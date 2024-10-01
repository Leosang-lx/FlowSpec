"""
Only for testing the distributed decoding performance
"""
import argparse
import sys

import torch
import torch.distributed as dist

from autoregressive_inference import load_local_pretrained_model, model_path
from cmd_util import *
from comm import *
from dist_comm.network_config import *

# parse server ip (optional) and group size
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ip', type=str, required=False, default=SERVER_IP)
parser.add_argument('-s', '--size', type=int, required=True)

arguments = parser.parse_args()
server_ip = arguments.ip
world_size = arguments.size


class DecodingWorker:
    """
    Worker for distributed LLM decoding
    """
    def __init__(self):
        self.server_addr = (server_ip, port_tcp)
        self.loop = asyncio.get_event_loop()

        # system initialization: share the n_device and allocate ranks to all devices
        self.ip = get_ip_addr(INTERFACE)
        # 0 means server, -1 means helping worker and uninitialized
        self.rank = -1
        self.n_device = world_size
        self.init_method = None

        # rank 0
        self.server_socket = None
        self.connections = None
        # other
        self.client_socket = None

        print('Initialize connection and process group...')
        try:
            if self.ip == server_ip:  # the device is the server/master
                self.rank = SERVER_RANK  # SERVER_RANK=0

                self.server_socket = socket.create_server((self.ip, 8848), family=socket.AF_INET)
                self.connections = accept_n_connections(self.server_socket, self.n_device - 1)  # ensure n accepted connections
                # allocate rank by the connections' order
                send_rank_tasks = [async_send_data(self.connections[i - 1], i) for i in range(1, self.n_device)]  # only transmit rank
                self.loop.run_until_complete(asyncio.gather(*send_rank_tasks))

            else:  # helping worker
                client_socket = socket.create_connection(self.server_addr)
                self.rank = recv_data(client_socket)
                # client_socket.close()  # 暂时先用自己的socket来传数据
        except Exception as e:
            print(e)
            sys.exit(0)

        # initial distributed group using torch.distributed
        assert isinstance(self.rank, int) and self.rank != -1
        print(f'  Rank={self.rank}')
        self.init_method = gen_init_method(server_ip, port_torch)

        try:
            dist.init_process_group(backend='gloo', init_method=self.init_method, world_size=self.n_device, rank=self.rank)
        except Exception as e:
            print(e)
            sys.exit(0)

        if dist.is_initialized():
            print('Succeed to initialize the process group')
        else:
            raise Exception('Fail to initialize the process group')

        # init model
        self.config, self.tokenizer, self.model = load_local_pretrained_model(model_path)
        self.model_config = (V, P, N, d_model, h, d_h, r) = (self.config.vocab_size, self.config.n_positions, self.config.n_layer,
                                                          self.config.n_embd, self.config.n_head, self.config.n_embd // self.config.n_head, 4)
        # init input and output configuration
        self.batch_size = 1
        self.text = "在一个风和日丽的下午，小镇的街道上人来人往，孩子们在巷口追逐嬉戏。李阿姨拿着刚从市场买回来的菜篮子，步履轻盈地走回家。街边的老槐树下，几位老人正围坐在一起下象棋，不时传来欢声笑语。今天是不是一个好日子？"
        self.input_length = len(self.text)
        self.max_length = 200
        # init inference mode
        self.distributed_method = 'TP'
        # TP
        self.h_split = 0
        # inference parameters
        self.use_cache = True
        self.do_sample = False
        self.top_k = 20
        self.top_p = 0.6

        self.KV_cache = None
        self.split_KV_cache = None

    def prepare_tp_decoding_send(self):
        """
        The rank_0 worker execute the prefill phase and transmit necessary data across the group
        :param past_key_values: tuple(N) * tuple(2) * tensor(b, seq, d_model)
        :param device_rank: 0_main worker; other_helping workers
        :return:
        """
        # if device_rank == server_rank:  # send necessary data to others
        V, P, N, d_model, h, d_h, r = self.model_config
        # check cache
        assert isinstance(self.KV_cache, tuple) and len(self.KV_cache) == N
        for layer_cache in self.KV_cache:
            assert isinstance(layer_cache, tuple) and len(layer_cache) == 2
            k_layer, v_layer = layer_cache
            assert isinstance(k_layer, torch.Tensor) and isinstance(v_layer, torch.Tensor)
            # assert k_layer.shape == v_layer.shape == (self.batch_size, h, input_length, d_h)

        # split cache by head: tuple(N) * tuple(2) * tensor(b, h, l_t, d_h) -> n * (list(N) * tuple(2) * tensor(b, h/n, l_t, d_h))
        assert self.config.n_head % self.n_device == 0
        h_split = h // self.n_device
        split_cache = [[] * self.n_device]
        # todo: 感觉不如改成逐个发，后面有可能会遇到内存紧张的情况，不过也可以后面再改
        for k_layer, v_layer in self.KV_cache:
            k_splits = [k_p.clone().detach() for k_p in torch.split(k_layer, self.n_device, dim=1)]
            v_splits = [v_p.clone().detach() for v_p in torch.split(v_layer, self.n_device, dim=1)]

            for i, kv_layer_split in enumerate(zip(k_splits, v_splits)):
                split_cache[i].append(kv_layer_split)
        self.split_KV_cache = split_cache[0]
        send_cache_tasks = [async_send_data(self.connections[rank-1], split_cache[rank]) for rank in range(1, self.n_device)]
        self.loop.run_until_complete(asyncio.gather(*send_cache_tasks))
        # return local_cache

    # else:  # accept data from rank_0 device

    def prepare_tp_decoding_recv(self):
        self.split_KV_cache = recv_data(self.client_socket)

    def TP_decoding(self):
        assert dist.is_initialized()
        if self.rank == SERVER_RANK:
            pass
        else:
            pass


