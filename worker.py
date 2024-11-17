"""
Only for testing the distributed decoding performance
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from cmd_util import get_ip_addr
from comm import *
from dist_comm.network_config import *
from forward_use_weight import *

# parse server ip (optional) and group size
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ip', type=str, required=False, default=MAIN_WORKER_IP)
parser.add_argument('-r', '--rank', type=int, required=False, default=0)
parser.add_argument('-s', '--size', type=int, required=False, default=DEFAULT_SIZE)

arguments = parser.parse_args()
server_ip = arguments.ip
rank = arguments.rank
world_size = arguments.size
# Enable when using RaspberryPi
if distributed:
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'


class DecodingWorker:
    """
    Worker for distributed LLM decoding
    """

    def __init__(self, main_worker_addr: tuple):
        # server_ip, port = server_addr
        self.server_addr = main_worker_addr
        self.loop = asyncio.get_event_loop()

        # system initialization: share the n_device and allocate ranks to all devices
        if distributed:
            self.ip = get_ip_addr(INTERFACE)
        else:
            self.ip = '127.0.0.1'
        print(f'Local ip: {self.ip}')
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
            if distributed:
                rank0 = self.ip == server_ip
                af = socket.AF_INET  # ipv4
            else:
                rank0 = rank == 0
                af = socket.AF_INET6

            if rank0:
                self.rank = SERVER_RANK  # SERVER_RANK=0
                self.server_socket = socket.create_server((server_ip, port_tcp), family=af)
                self.connections = accept_n_connections(self.server_socket,
                                                        self.n_device - 1)  # ensure n accepted connections
                # allocate rank by the connections' order
                send_rank_tasks = [async_send_data(self.connections[i - 1][0], i) for i in
                                   range(1, self.n_device)]  # only transmit rank

                self.loop.run_until_complete(asyncio.gather(*send_rank_tasks))

            else:  # helping worker
                self.client_socket = socket.create_connection(self.server_addr)
                # self.client_socket = socket.create_connection(('127.0.0.1', port_tcp))
                self.rank = recv_data(self.client_socket)
                # self.client_socket.close()  # 暂时先用自己的socket来传数据
        except Exception as e:
            print(e)
            sys.exit(0)

        # initial distributed group using torch.distributed
        assert isinstance(self.rank, int) and self.rank != -1
        print(f'  Rank={self.rank}')
        self.init_method = gen_init_method(server_ip, port_torch)
        print(f'init_method={self.init_method}')

        try:
            dist.init_process_group(backend='gloo', init_method=self.init_method, world_size=self.n_device,
                                    rank=self.rank)
        except Exception as e:
            print(e)
            sys.exit(0)

        if dist.is_initialized():
            print(f'Succeed to initialize the process group with rank={self.rank}')
        else:
            raise Exception('Fail to initialize the process group')

        """
        weights (split):
            token embedding, position embedding
        weights layers (split): *12
                QKV projection, LayerNorm, MLP layer, LayerNorm
        lm_head:
            token embedding or independent weight (d_model, vocab_size)
        """
        self.split_weights = None
        self.layers_MLP_weights = None
        self.MLP_activation = transformers.activations.NewGELUActivation()

        self.tokenizer = None
        self.config = None

        # self.embedding_weights = None
        self.token_embedding = None
        self.position_embedding = None
        self.layers_ln_weight = None
        self.all_split_embeddings = None
        self.split_heads = None  # split heads range (indexes set from 0 to h-1)
        self.split_embedding_range = None  # = split_heads * d_h

        self.ln_f_weight = None
        self.lm_head = None

        self.split_KV_cache = None

        # init input and output configuration
        # self.tokenizer = load_local_pretrained_model(model_path, 'tokenizer')[0]
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

    def prepare_cache_tp_send(self):
        """
        The rank_0 worker execute the prefill phase and transmit necessary data across the group
        :param past_key_values: tuple(N) * tuple(2) * tensor(b, seq, d_model)
        :param device_rank: 0_main worker; other_helping workers
        :return:
        """
        # if device_rank == server_rank:  # send necessary data to others
        V, P, N, d_model, h, d_h, r = self.config
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
        send_cache_tasks = [async_send_data(self.connections[rank - 1], split_cache[rank]) for rank in
                            range(1, self.n_device)]
        self.loop.run_until_complete(asyncio.gather(*send_cache_tasks))
        # return local_cache

    # else:  # accept data from rank_0 device

    def tp_init(self, conn_to_master: socket.socket, split_MLP=True):
        """
        init necessary weights for tensor parallelism
        :param conn_to_master:
        :param split_MLP:
        :return:
        """
        request = f'TP_WEIGHT {self.rank} SPLIT_MLP {int(split_MLP)}\n'
        conn_to_master.sendall(request.encode())
        data = recv_data(conn_to_master)
        if self.rank == 0:
            if not split_MLP:
                self.config, self.split_weights, self.tokenizer, embedding_weights, self.ln_f_weight, \
                    self.layers_MLP_weights = data
            else:
                self.config, self.split_weights, self.tokenizer, embedding_weights, self.ln_f_weight = data

            token_embedding_weight, position_embedding_weight = embedding_weights
            self.token_embedding = nn.Embedding.from_pretrained(token_embedding_weight)
            self.position_embedding = nn.Embedding.from_pretrained(position_embedding_weight)
            self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(token_embedding_weight)

        else:  # other workers
            self.config, self.split_weights = data

    def se_init(self, conn_to_master: socket.socket):
        request = f'SE_WEIGHT {self.rank}\n'
        conn_to_master.sendall(request.encode())
        data = recv_data(conn_to_master)
        if self.rank == 0:
            self.config, self.split_weights, self.tokenizer, embedding_weights, self.all_split_embeddings = data

            token_embedding_weight, position_embedding_weight = embedding_weights
            self.token_embedding = nn.Embedding.from_pretrained(token_embedding_weight)
            self.position_embedding = nn.Embedding.from_pretrained(position_embedding_weight)
            self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(token_embedding_weight)
        else:
            self.config, self.split_weights = data

        h_d = self.config.h // self.n_device
        self.split_heads = self.rank * h_d, (self.rank + 1) * h_d  # default: equal split
        h_l, h_r = self.split_heads
        self.split_embedding_range = h_l * self.config.d_h, h_r * self.config.d_h  # default: equal split

    def init(self, split_MLP=True, split_embedding=False):
        """
        Receive split_weight and config from master
        """
        with socket.create_connection((MASTER_IP, master_port)) as conn:
            if self.rank == 0:  # main worker
                print('Send WORKER_NUM to master')
                request = f'WORKER_NUM {self.n_device}\n'
                conn.sendall(request.encode())
                reply = recv_data(conn)
                if reply != 'OK':
                    raise Exception('Unknown reply from master')
                reply_tasks = [async_send_data(self.connections[rank - 1][0], 'OK') for rank in
                               range(1, self.n_device)]
                self.loop.run_until_complete(asyncio.gather(*reply_tasks))
            else:
                reply = recv_data(self.client_socket)
                if reply != 'OK':
                    raise Exception('Unknown reply from main worker')

            # ask necessary weights from master
            print('Get necessary weights from master')
            if split_embedding:
                self.se_init(conn)
            else:
                self.tp_init(conn, split_MLP)

        print('TP Init Done!')

    def co_forward_tp(self, input_ids=None, use_cache=True, split_MLP=True):
        """
        :param input_ids: only the rank-0 device needs input
        :param use_cache: use KV-cache to reduce duplicated computation
        :return:
        """
        # split_heads = self.config.split_heads
        d_model = self.config.d_model

        # latency_measure
        all_reduce_latency = []

        if self.split_KV_cache is None:
            past_length = 0
            self.split_KV_cache = (None,) * self.config.n_layer
        else:
            past_length = self.split_KV_cache[0][0].size(-2)
        # print(f'past_length={past_length}')
        if use_cache:
            cache_present = ()  # store updated KV-Cache by layer

        if self.rank == 0:  # main worker
            token_embedding = self.token_embedding(input_ids)
            position_ids = torch.arange(past_length, input_ids.shape[-1] + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)
            position_embedding = self.position_embedding(position_ids)
            hidden_states = token_embedding + position_embedding
            input_shape = torch.tensor(hidden_states.shape, dtype=torch.int32)
            dist.broadcast(input_shape, src=0)
            # send_shapes = [async_send_data(conn, tuple(hidden_states.shape)) for conn, _ in self.connections]
            # self.loop.run_until_complete(asyncio.gather(*send_shapes))  # send shape

        else:
            # send shape first
            input_shape = torch.zeros(3, dtype=torch.int32)
            dist.broadcast(input_shape, src=0)
            # input_shape = recv_data(self.client_socket)
            hidden_states = torch.zeros(*input_shape)

        dist.broadcast(hidden_states, src=0)  # init input

        # forward_layers
        for layer_idx in range(self.config.n_layer):
            split_weight_layer = self.split_weights[layer_idx]
            ln1_w_b, ln2_w_b = split_weight_layer['LN']
            split_cache_layer = self.split_KV_cache[layer_idx]

            residual = hidden_states
            # LN1
            hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, self.config.ln_eps)

            # split MHA
            split_MHA_weight = split_weight_layer['MHA']
            split_MHA_output, layer_cache = MHA_forward_use_weights(hidden_states, split_MHA_weight, self.config,
                                                                    split_cache_layer)
            if use_cache:
                cache_present = cache_present + (layer_cache,)

            if split_MLP:  # split MLP inference on workers
                # AllReduce
                start_a_r = time.perf_counter()
                dist.all_reduce(split_MHA_output, op=dist.ReduceOp.SUM)
                all_reduce_latency.append(time.perf_counter() - start_a_r)

                # Residual connection
                hidden_states = split_MHA_output + residual

                if self.rank == 0:  # central
                    residual = hidden_states
                # LN2
                hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, self.config.ln_eps)
                # split MLP
                split_MLP_weight = split_weight_layer['MLP']
                split_MLP_output = MLP_forward_use_weights(hidden_states, split_MLP_weight, self.config)

                # AllReduce or SingleReduce (for the last layer)
                start_a_r = time.perf_counter()
                if layer_idx == self.config.n_layer - 1:
                    dist.reduce(split_MLP_output, dst=0, op=dist.ReduceOp.SUM)
                else:
                    dist.all_reduce(split_MLP_output, op=dist.ReduceOp.SUM)
                all_reduce_latency.append(time.perf_counter() - start_a_r)

                # Residual connection
                hidden_states = split_MLP_output + residual

            else:  # inference MLP only on device 0
                dist.reduce(split_MHA_output, dst=0, op=dist.ReduceOp.SUM)
                if self.rank == 0:
                    hidden_states = split_MHA_output + residual
                    residual = hidden_states
                    # LN2
                    hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, self.config.ln_eps)
                    # MLP
                    MLP_weight_layer = self.layers_MLP_weights[layer_idx]
                    hidden_states = MLP_forward_use_weights(hidden_states, MLP_weight_layer, self.config)
                    # Residual connection
                    hidden_states = hidden_states + residual

        if use_cache:  # udpate cache
            self.split_KV_cache = cache_present

        print(f'All-Reduce: total {np.sum(all_reduce_latency)}s, avg {np.mean(all_reduce_latency)}s')

        if self.rank == 0:
            hidden_states = F.layer_norm(hidden_states, (d_model,), *self.ln_f_weight, self.config.ln_eps)
            return hidden_states
            # else:  # other worker: 返回当前长度
            #     if use_cache:
            #         return self.split_KV_cache[0][0].size(-2) + 1
            #     else:
            #         return hidden_states.size(-2)

    def co_forward_se(self, input_ids=None, use_cache=True):
        """
        :param input_ids: only the rank-0 device needs input
        :param use_cache: use KV-cache to reduce duplicated computation
        :return:
        """
        # split_heads = self.config.split_heads
        d_model = self.config.d_model
        d_h = self.config.d_h

        distr_ln_latency = []

        # init cache
        if self.split_KV_cache is None:
            past_length = 0
            self.split_KV_cache = (None,) * self.config.n_layer
        else:
            past_length = self.split_KV_cache[0][0].size(-2)

        if use_cache:
            cache_present = ()  # store updated KV-Cache by layer

        # init inference
        # start_init = time.perf_counter()
        if self.rank == 0:  # main worker todo: choose to send tokens or hidden_states
            token_embedding = self.token_embedding(input_ids)
            position_ids = torch.arange(past_length, input_ids.shape[-1] + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)
            position_embedding = self.position_embedding(position_ids)
            hidden_states = token_embedding + position_embedding
            input_shape = torch.tensor(hidden_states.shape, dtype=torch.int32)
            dist.broadcast(input_shape, src=0)

        else:
            # send shape first
            input_shape = torch.zeros(3, dtype=torch.int32)
            dist.broadcast(input_shape, src=0)
            hidden_states = torch.zeros(*input_shape)

        dist.broadcast(hidden_states, src=0)  # init hidden_states
        # end_init = time.perf_counter()
        # print(f'{end_init - start_init}s for inference init')

        # forward layers
        for layer_idx, (layer_weights, layer_cache) in enumerate(zip(self.split_weights, self.split_KV_cache)):
            ln1_w_b, ln2_w_b = layer_weights['LN']

            # MHA{
            QKV_proj_w_b, Wo_proj_w_b = layer_weights['MHA']
            # merge QKV_proj_w_b
            if len(QKV_proj_w_b) == 3:
                QKV_proj_w_b = tuple([torch.concat(ws_or_bs, dim=-1) for ws_or_bs in zip(*QKV_proj_w_b)])

            if layer_idx == 0:  # 1st layer
                # residual 1
                se_l, se_r = self.split_embedding_range
                residual = hidden_states[..., se_l:se_r]
                # LN1: complete weight
                hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, self.config.ln_eps)

                # QKV projection
                (Q, K, V), layer_cache = QKV_proj(hidden_states, QKV_proj_w_b, self.config.d_h, layer_cache)

            else:  # 2-n_l layers
                # residual 1
                residual = split_embeddings

                # LN1: split_embedding
                start_ln = time.perf_counter()
                split_embeddings = distributed_layer_norm(split_embeddings, ln1_w_b, d_model, self.config.ln_eps)
                distr_ln_latency.append(time.perf_counter() - start_ln)

                # QKV projection: split_embedding
                QKV = self.ring_gather_reduce_comp_overlap(split_embeddings, *QKV_proj_w_b)

                Q, K, V = split_QKV(QKV, self.config.d_h)
                if layer_cache is not None:  # update layer_cache
                    K_cache, V_cache = layer_cache
                    K = torch.cat((K_cache, K), dim=2)
                    V = torch.cat((V_cache, V), dim=2)
                layer_cache = (K, V)

            if use_cache:  # update cache
                cache_present = cache_present + (layer_cache,)

            # attn
            attn_output, _ = attn(Q, K, V)
            split_embeddings = merge_heads(attn_output, d_h)

            # Wo projection
            split_embeddings = self.ring_reduce_scatter_comp_overlap(split_embeddings, *Wo_proj_w_b)

            # residual connection 1
            split_embeddings += residual
            # }MHA

            # residual 2
            residual = split_embeddings

            # MLP
            mlp1_w_b, mlp2_w_b = layer_weights['MLP']

            # LN2: split_embedding
            start_ln = time.perf_counter()
            split_embeddings = distributed_layer_norm(split_embeddings, ln2_w_b, d_model, self.config.ln_eps)
            distr_ln_latency.append(time.perf_counter() - start_ln)

            # MLP1
            split_embeddings = self.ring_gather_reduce_comp_overlap(split_embeddings, *mlp1_w_b)

            # activation
            split_embeddings = self.MLP_activation(split_embeddings)

            if layer_idx != self.config.n_layer - 1:
                # MLP2
                split_embeddings = self.ring_reduce_scatter_comp_overlap(split_embeddings, *mlp2_w_b)

                # residual connection 2
                split_embeddings = residual + split_embeddings

            else:  # final layer
                # start all-gather the residual of split_embeddings
                # init tensor container according to embedding size of all devices
                residual_container = [torch.empty(*split_embeddings.shape[:-1], split_embeddings_size)
                                      for split_embeddings_size in
                                      self.all_split_embeddings] if self.rank == 0 else None
                residual_gather_task = dist.gather(residual, residual_container, dst=0, async_op=True)
                # MLP2
                hidden_states = Conv1D_forward_use_weights(split_embeddings, mlp2_w_b)

                # reduce to rank0 device
                dist.reduce(hidden_states, dst=0)

                residual_gather_task.wait()

        if use_cache:  # udpate cache
            self.split_KV_cache = cache_present

        print(f'Distributed LayerNorm: total {np.sum(distr_ln_latency)}s, avg {np.mean(distr_ln_latency)}s')

        if self.rank == 0:
            residual = torch.concat(residual_container, dim=-1)
            # final residual connection 2
            hidden_states += residual
            return hidden_states

    def tp_generate(self, max_length, split_MLP=True, split_embedding=False):
        if split_embedding:
            tp_forward = self.co_forward_se
        else:
            tp_forward = self.co_forward_tp

        # prefill
        if self.rank == 0:
            input_ids = torch.LongTensor([self.tokenizer.convert_tokens_to_ids(list(self.text))])
            cur_len = input_ids.size(-1)
            assert cur_len < max_length
            # share the cur_len
            input_size = torch.tensor(input_ids.shape, dtype=torch.int32)
            dist.broadcast(input_size, src=0)

            self.split_KV_cache = None  # clear cache
            generated_text = []

            print('Start Prefill')
            start_prefill = time.perf_counter()
            hidden_states = tp_forward(input_ids)

            start_lmhead = time.perf_counter()
            logits = self.lm_head(hidden_states[:, -1, :])
            next_token_ids = logits2token(logits)
            input_ids = next_token_ids
            end_prefill = time.perf_counter()
            print(f'{end_prefill - start_lmhead}s for lm_head and token generation')

        else:  # other workers
            input_size = torch.empty(2, dtype=torch.int32)
            dist.broadcast(input_size, src=0)
            cur_len = input_size[-1]
            print('Start Prefill')
            tp_forward()

        cur_len += 1

        # decoding
        print('Start decoding')
        start_decoding = time.perf_counter()
        for seq_len in tqdm(range(cur_len, max_length + 1)):
            if self.rank == 0:
                hidden_states = tp_forward(input_ids)
                logits = self.lm_head(hidden_states[:, -1, :])
                next_token_ids = logits2token(logits)
                generated_token = self.tokenizer.convert_ids_to_tokens(next_token_ids)
                generated_text.append(generated_token[0])
                # input_ids = torch.concat((input_ids, next_token_ids), dim=-1)
                input_ids = next_token_ids
                # next_tokens = self.tokenizer.convert_ids_to_tokens
            else:
                tp_forward()
        end_decoding = time.perf_counter()

        if self.rank == 0:
            prefill_t = end_prefill - start_prefill
            decoding_t = end_decoding - start_decoding
            print(f'Prefill : {prefill_t}s')
            print(f'Decoding: {decoding_t}s')
            return ''.join(generated_text)

    def get_ring_index(self, index: int, offset: int):  # ring index starts from 0
        ring_index = index + offset
        if ring_index < 0:
            ring_index += self.n_device
        elif ring_index >= self.n_device:
            ring_index %= self.n_device
        return ring_index

    def ring_reduce_scatter_comp_overlap(self, xi: torch.Tensor, Wi: torch.Tensor, bi=None):
        """
        overlap the ring reduce-scatter comm operation with the **previous** matrix-vector multiplication
        :return: split reduced results
        """
        W_in, W_out = Wi.shape
        # the whole weight matrix is split along row dimension(dim=0) across devices
        assert xi.size(-1) == W_in and W_out % self.n_device == 0

        # further split partial weight along column dimension(dim=1)
        W_split_size = W_out // self.n_device
        Wi_split = Wi.split(W_split_size, dim=-1)

        comm_from_index = self.get_ring_index(self.rank, -1)
        from_tensor = torch.zeros(*xi.shape[:-1], W_split_size)
        comm_to_index = self.get_ring_index(self.rank, 1)

        # split_results = [None] * self.n_device

        for i in range(self.n_device - 1, -1, -1):  # reverse order from n-1 to 0

            comp_index = self.get_ring_index(self.rank, i)  # [rank-1, rank-2, ..., rank]
            # start receiving
            if i != self.n_device - 1:
                send_task = dist.isend(yi, dst=comm_to_index)
                # print(f'd_{self.rank} send to d_{comm_to_index}')
                # print(f'{self.rank} recv from d_{comm_from_index}...', end='')
                recv_task = dist.irecv(from_tensor, src=comm_from_index)

            else:
                recv_task = None

            # partial computation
            yi = torch.matmul(xi, Wi_split[comp_index])

            # recv result from others
            if recv_task:
                send_task.wait()
                recv_task.wait()
                # print(f'done')
                yi.add_(from_tensor)  # reduce
        if bi is not None:
            yi.add_(bi)
        return yi

    def ring_all_gather_comp_overlap(self, xi: torch.Tensor, Wi: torch.Tensor, b=None):
        # 好像用不着，先留着吧，考虑作为benchmark的时候用
        """
        overlap the ring all-gather comm operation with the **following** matrix-vector multiplication
        :return: gathered results after the matrix-vector multiplication
        """
        W_in, W_out = Wi.shape
        assert xi.size(-1) == W_in and W_out % self.n_device == 0

        comm_from_index = self.get_ring_index(self.rank, -1)
        from_tensor = torch.zeros_like(xi)  # from_tensor for saving the received tensor
        to_tensor = xi  # to_tensor for comp and comm at each turn
        comm_to_index = self.get_ring_index(self.rank, 1)

        split_results = [None] * self.n_device

        for i in range(self.n_device - 1, -1, -1):  # reverse order from n-1 to 0
            comp_index = self.get_ring_index(self.rank, i)  # shift order according to rank and i
            if i > 0:
                # start sending
                send_task = dist.isend(to_tensor, dst=comm_to_index)
                recv_task = dist.irecv(from_tensor, src=comm_from_index)

            # partial computation
            split_results[comp_index] = to_tensor @ Wi

            if i > 0:
                # send_task.wait()
                recv_task.wait(timeout=timeout_max)
                # exchange reference after send_task and recv_task are both finished
                from_tensor, to_tensor = to_tensor, from_tensor
            else:
                assert None not in split_results
                return split_results

    def ring_gather_reduce_comp_overlap(self, xi: torch.Tensor, Wi: torch.Tensor, bi=None):
        """
        overlap the ring all-gather comm operation with the **following** matrix-vector multiplication
        introduce additional reduce for splitting the embedding dimension
        :return:
        """
        W_in, W_out = Wi.shape
        # the whole weight matrix is split along column dimension(dim=1) across devices
        assert W_in % self.n_device == 0 and xi.size(-1) == (W_in // self.n_device)
        # further split partial weight along row dimension(dim=0)
        W_split_size = W_in // self.n_device
        Wi_split = Wi.split(W_split_size, dim=0)

        # the last index and next index of ring structure
        comm_from_index = self.get_ring_index(self.rank, -1)
        comm_to_index = self.get_ring_index(self.rank, 1)
        from_tensor = torch.zeros_like(xi)
        # to_tensor = xi
        y_reduce = None

        for i in range(self.n_device - 1, -1, -1):  # reverse order from n-1 to 0
            if i > 0:
                send_task = dist.isend(xi, dst=comm_to_index)  # send_task should be kept, but don't wait
                recv_task = dist.irecv(from_tensor, src=comm_from_index)

            comp_index = self.get_ring_index(self.rank, i + 1)  # [rank, rank-1, ..., rank+1]

            yi = torch.matmul(xi, Wi_split[comp_index])  # mm
            if y_reduce is not None:
                y_reduce.add_(yi)
            else:
                y_reduce = yi

            if i > 0:
                send_task.wait()
                recv_task.wait()  # recv next xi from other device
                # send_task.wait(timeout=timeout_max)
                xi, from_tensor = from_tensor, xi  # exchange reference
            else:
                if bi is not None:
                    y_reduce.add_(bi)
                return y_reduce


def distributed_layer_norm(split_embedding, split_LN_weights, d_model, eps):
    x_sum = split_embedding.sum(dim=-1, keepdim=True)
    # obtain x.SUM
    dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
    x_mean = x_sum.div(d_model)
    x_ssd = (split_embedding - x_mean).square().sum(dim=-1, keepdim=True)
    # obtain x.SSD (Sum of Squared Deviations)
    dist.all_reduce(x_ssd, op=dist.ReduceOp.SUM)
    x_var = x_ssd.div(d_model)

    # perform partial LayerNorm on split_embedding
    split_embedding = (split_embedding - x_mean) / torch.sqrt(x_var + eps)
    if split_LN_weights is not None:
        split_w, split_b = split_LN_weights
        return split_embedding * split_w + split_b
    return split_embedding


def test_overlap(w: DecodingWorker):
    # test overlap
    b = 1
    d_model = 4096
    intermediate_dimension = 11008
    dtype = torch.float32
    split_size = d_model // world_size
    xi = torch.randn(split_size, dtype=dtype)
    print(xi)

    # test ring_reduce_scatter_overlap()
    Wi = torch.randn(split_size, d_model, dtype=dtype)
    print(Wi)
    yi = w.ring_reduce_scatter_comp_overlap(xi, Wi)
    print(yi)
    # verify result
    xis = [torch.empty_like(xi, dtype=dtype)] * world_size
    dist.all_gather(xis, xi)
    for x in xis:
        print(torch.equal(x, xi))
    x = torch.concat(xis, dim=0)
    Wis = [torch.empty_like(Wi, dtype=dtype)] * world_size
    dist.all_gather(Wis, Wi)  # bug
    Wis[rank] = Wi
    Wis = torch.concat(Wis, dim=0)
    print(Wis)
    Wi = torch.split(Wis, split_size, dim=-1)[rank]
    yi_correct = x @ Wi
    print(torch.allclose(yi_correct, yi, atol=1e-3))  # >> True
    print(torch.equal(yi_correct, yi))  # >> False
    print(yi_correct)

    # test ring_reduce_scatter_overlap()
    Wi = torch.randn(d_model, intermediate_dimension // world_size, dtype=dtype)
    print(Wi)
    yi = w.ring_gather_reduce_comp_overlap(xi, Wi)
    print(yi)
    # verify result
    xis = [torch.empty_like(xi, dtype=dtype)] * world_size
    dist.all_gather(xis, xi)  # bug
    xis[rank] = xi
    x = torch.concat(xis, dim=0)
    yi_correct = x @ Wi
    print(torch.allclose(yi_correct, yi, atol=1e-3))  # >> True
    print(torch.equal(yi_correct, yi))  # >> False
    print(yi_correct)


if __name__ == '__main__':
    worker = DecodingWorker((MAIN_WORKER_IP, port_tcp))

    # test TP
    params = {'split_embedding': True}

    worker.init(**params)
    print(worker.text)
    generated_text = worker.tp_generate(105, **params)
    if worker.rank == 0:
        print(generated_text)

    # test overlap
    # test_overlap(worker)

    # test split_embedding
