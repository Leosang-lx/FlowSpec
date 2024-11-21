import asyncio
from dist_comm.network_config import *
from autoregressive_inference import model_path, load_local_pretrained_model
from types import SimpleNamespace
from forward_use_weight import get_transformer_model_weight
from distributedTP import split_weight_TP
from comm import *


class Master:
    def __init__(self):
        # self.main_worker_ip = MAIN_WORKER_IP
        self.server_ip = '192.168.1.150'

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

        self.server = None

    async def handle_worker(self, reader, writer):
        client_addr = writer.get_extra_info('peername')
        try:
            while True:
                request = await reader.readline()
                request = request.decode().strip()
                print(f'{client_addr}: {request}')
                fields = request.split()
                # if len(fields) == 2 and fields[1].isdigit():
                if fields[0] == 'WORKER_NUM':
                    n_workers = int(fields[1])
                    assert 1 < n_workers < 11
                    self.n_workers = n_workers
                    self.split_weights = None
                    self.split_heads = [self.model_config.h // self.n_workers] * self.n_workers
                    writer.write(gen_bytes('OK'))
                    await writer.drain()
                else:
                    if fields[0] == 'TP_WEIGHT':  # send model weights
                        worker_rank = int(fields[1])
                        assert self.n_workers > 0 and worker_rank < self.n_workers

                        if len(fields) == 4 and fields[2] == 'SPLIT_MLP':
                            split_mlp = bool(int(fields[3]))
                        else:
                            split_mlp = True

                        if self.split_weights is None or self.split_embedding:
                            self.split_weights = split_weight_TP(self.model_weights, self.n_workers,
                                                                 self.model_config, split_MLP=split_mlp)  # default: equal split
                            self.split_embedding = False

                        # ln_weights = tuple(
                        #     [layer_weight['LN'] for layer_weight in self.model_weights['layers_weights']])
                        ln_f_weights = self.model_weights['ln_f_weights']
                        # ln_weights = ln_weights, ln_f_weights
                        data_to_send = self.model_config, self.split_weights[worker_rank]

                        if worker_rank == 0:  # additional embedding weights and layer norm weights for central processing
                            data_to_send = data_to_send + (
                                self.tokenizer, self.model_weights['embedding_weights'], ln_f_weights)

                    elif fields[0] == 'SE_WEIGHT':  # send model weights: split_embedding=True
                        worker_rank = int(fields[1])
                        assert self.n_workers > 0 and worker_rank < self.n_workers
                        if self.split_weights is None or not self.split_embedding:
                            self.split_weights = split_weight_TP(self.model_weights, self.n_workers, self.model_config,
                                                                 split_embedding=True)
                            self.split_embedding = True

                        ln_f_weights = self.model_weights['ln_f_weights']
                        data_to_send = self.model_config, self.split_weights[worker_rank]
                        if worker_rank == 0:
                            # all_split_embeddings = [
                            #                            self.model_config.d_model // self.n_workers] * self.n_workers  # equal split
                            data_to_send = data_to_send + (
                                self.tokenizer, self.model_weights['embedding_weights'], ln_f_weights)
                    else:
                        raise Exception(f'Unknown request: {request}')

                    writer.write(gen_bytes(data_to_send))
                    await writer.drain()
                    break

        except asyncio.CancelledError or ConnectionResetError as e:
            print(f"Main connection closed to worker {client_addr}", e)
        finally:
            writer.close()
            await writer.wait_closed()

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
