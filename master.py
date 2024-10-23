import asyncio
from dist_comm.network_config import *
from autoregressive_inference import *
from distributedTP import *
from comm import *


class Master:
    def __init__(self):
        # self.main_worker_ip = MAIN_WORKER_IP
        self.server_ip = '192.168.1.150'

        self.n_workers = 0
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
        self.split_weights = None

        self.server = None

    async def handle_worker(self, reader, writer):
        client_addr = writer.get_extra_info('peername')
        try:
            while True:
                data = await reader.readline()
                data = data.decode().strip()
                print(f'{client_addr}: {data}')
                fields = data.split()
                # if len(fields) == 2 and fields[1].isdigit():
                if fields[0] == 'WORKER_NUM':
                    n_workers = int(fields[1])
                    assert 1 < n_workers < 11
                    self.n_workers = n_workers
                    self.split_heads = [self.model_config.h // self.n_workers] * self.n_workers
                    writer.write(gen_bytes('OK'))
                    await writer.drain()

                elif fields[0] == 'TP_WEIGHT':  # send model weights
                    worker_rank = int(fields[1])
                    assert self.n_workers > 0 and worker_rank < self.n_workers
                    if self.split_weights is None:
                        self.split_weights = split_weight_TP(self.model_weights, self.model_config.h,
                                                             self.n_workers, self.model_config)

                    ln_weights = tuple([layer_weight['LN'] for layer_weight in self.model_weights['layers_weights']])
                    ln_f_weights = self.model_weights['ln_f_weights']
                    ln_weights = ln_weights, ln_f_weights
                    data_to_send = self.model_config, self.tokenizer, self.split_weights[worker_rank], ln_weights
                    if worker_rank == 0:  # embedding weights and layer norm weights for central processing
                        embedding_weights = self.model_weights['embedding_weights']
                        data_to_send = data_to_send + (embedding_weights,)
                        if len(fields) == 4 and fields[2] == 'SPLIT_MLP':
                            split_mlp = bool(int(fields[3]))
                            if not split_mlp:
                                layers_MLP_weights = [layer_weights['MLP'] for layer_weights in self.model_weights['layers_weights']]
                                data_to_send = data_to_send + (layers_MLP_weights,)

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
