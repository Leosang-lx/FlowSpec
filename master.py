import asyncio

import torch

from dist_comm.network_config import *
from autoregressive_inference import model_path, load_local_pretrained_model, logits2token
from types import SimpleNamespace
from forward_use_weight import get_transformer_model_weight
from distributedTP import split_weight_TP
from comm import *
from sampling import apply_sampling


class Master:
    def __init__(self):
        # self.main_worker_ip = MAIN_WORKER_IP
        self.server_ip = MASTER_IP

        self.n_workers = 0
        print('Load model from', model_path)
        self.config, self.tokenizer, self.model = load_local_pretrained_model(model_path)
        config = self.config
        vocab_size, max_p, n_layer, d_model, h, d_h, rate = \
            (
                config.vocab_size, config.n_positions, config.n_layer, config.n_embd, config.n_head,
                config.n_embd // config.n_head,
                4)
        layer_norm_eps = config.layer_norm_epsilon
        # dropout_prob = config.resid_pdrop
        dropout_prob = 0
        model_config = {
            'vocab_size': vocab_size,
            'max_position': max_p,
            'n_layer': n_layer,
            'd_model': d_model,
            'h': h,
            'd_h': d_h,
            'rate': rate,
            'ln_eps': layer_norm_eps,
            'dropout_prob': dropout_prob
        }
        self.model_config = SimpleNamespace(**model_config)

        self.model_weights = get_transformer_model_weight(self.model.transformer)
        self.split_heads = None
        self.split_embedding = None
        self.split_weights = None
        self.asyncio_event = asyncio.Event()
        self.done_workers = 0

        self.server = None

    async def handle_worker(self, reader, writer):
        client_addr = writer.get_extra_info('peername')
        try:
            while True:
                request = await reader.readline()
                request = request.decode().strip()
                print(f'{client_addr}: {request}')
                fields = request.split()

                # if fields[0] == 'WORKER_NUM':
                #     n_workers = int(fields[1])
                #     assert 1 < n_workers < 11
                #     self.n_workers = n_workers
                #     self.split_weights = None
                #     # self.split_heads = [self.model_config.h // self.n_workers] * self.n_workers
                #     writer.write(gen_bytes('OK'))
                #     await writer.drain()
                if fields[0] in ('TP_WEIGHT', 'SE_WEIGHT', 'TP_CACHE'):
                    worker_rank = int(fields[1])
                    if worker_rank != 0:
                        await self.asyncio_event.wait()
                    else:
                        self.n_workers = int(fields[2])

                    if fields[0] == 'TP_CACHE':
                        input_text = fields[3]
                        input_ids = torch.LongTensor([self.tokenizer.convert_tokens_to_ids(list(input_text))])
                        output = self.model(input_ids)
                        cache = output.past_key_values
                        self.split_cache_n = split_cache_by_head(cache)
                        writer.write(gen_bytes(cache))
                        await writer.drain()
                        hidden_states = output.last_hidden_states
                        logits = self.model.lm_head(hidden_states)
                        next_input_ids = logits2token(logits)
                        writer.write(next_input_ids)
                        await writer.drain()
                    else:
                        assert self.n_workers > 0 and worker_rank < self.n_workers
                        if worker_rank == 0:  # ensure the split_weights
                            if fields[0] == 'TP_WEIGHT':  # TP: Tensor Parallelism
                                if len(fields) == 4 and fields[2] == 'SPLIT_MLP':
                                    split_mlp = bool(int(fields[3]))
                                else:
                                    split_mlp = True

                                if self.split_weights is None or self.split_embedding:
                                    self.split_weights = split_weight_TP(self.model_weights, self.n_workers, self.model_config,
                                                                         split_MLP=split_mlp,
                                                                         return_view=True)  # default: equal split
                                    self.split_embedding = False

                            else:  # SE: split-embedding
                                if self.split_weights is None or not self.split_embedding:
                                    self.split_weights = split_weight_TP(self.model_weights, self.n_workers,
                                                                         self.model_config,
                                                                         split_embedding=True, return_view=True)
                                    self.split_embedding = True

                            self.asyncio_event.set()  # split weights have been ensured, go
                        await self.send_split_weights(writer, worker_rank)  # send split weights

                else:
                    raise Exception(f'Unknown request: {request}')

        except asyncio.CancelledError or ConnectionResetError as e:
            print(f"Main connection closed to worker {client_addr}", e)
        finally:
            writer.close()
            await writer.wait_closed()
            self.asyncio_event = asyncio.Event()

    async def send_split_weights(self, writer, rank):
        assert self.n_workers > 1
        # model_config
        config = self.model_config
        writer.write(gen_bytes(config))
        await writer.drain()
        # embedding
        embedding_weights = self.model_weights['embedding_weights']
        writer.write(gen_bytes(embedding_weights))
        await writer.drain()
        # split_weights: send by layers
        split_layers_weights = self.split_weights[rank]  # view of split_weights
        for split_layer_weights in split_layers_weights:
            blocks = split_layer_weights.keys()
            writer.write(gen_bytes(list(blocks)))
            await writer.drain()
            for block in blocks:  # ['MHA', 'MLP', 'LN']
                (proj1_w, proj1_b), (proj2_w, proj2_b) = split_layer_weights[block]
                if proj1_w is not None:
                    proj1_w = proj1_w.clone().contiguous()
                if proj1_b is not None:
                    proj1_b = proj1_b.clone().contiguous()
                if proj2_w is not None:
                    proj2_w = proj2_w.clone().contiguous()
                if proj2_b is not None:
                    proj2_b = proj2_b.clone().contiguous()
                block_weights = (proj1_w, proj1_b), (proj2_w, proj2_b)
                writer.write(gen_bytes(block_weights))
                await writer.drain()
        if rank == 0:
            # tokenizer
            writer.write(gen_bytes(self.tokenizer))
            await writer.drain()
            # ln_f
            ln_f_weights = self.model_weights['ln_f_weights']
            writer.write(gen_bytes(ln_f_weights))
            await writer.drain()

    async def start_handling(self):
        server = await asyncio.start_server(self.handle_worker, self.server_ip, master_port)
        addr = server.sockets[0].getsockname()
        print(f'Handling workers on {addr}')

        self.server = server
        async with server:
            await server.serve_forever()

    def start(self):
        asyncio.run(self.start_handling())



if __name__ == '__main__':
    m = Master()
    m.start()
