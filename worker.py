"""
Worker.py: distributed autoregressive inference for transformer-based LM
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime

# from memory_profiler import profile

from cmd_util import get_ip_addr
from comm import *
from dist_comm.network_config import *
from forward_use_weight import *


def get_timestamp():
    return datetime.datetime.now().strftime('%H:%M:%S')


# parse server ip (optional) and group size
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--addr', type=str, required=False, default=MAIN_WORKER_IP)
parser.add_argument('-i', '--rank', type=int, required=False, default=0)
parser.add_argument('-w', '--world_size', type=int, required=False, default=DEFAULT_SIZE)
parser.add_argument('-r', '--repeat', type=int, required=False, default=1)
parser.add_argument('-s', '--se', type=int, required=False, default=1)
parser.add_argument('-g', '--generate_length', type=int, required=False, default=2)  # default: prefill:1 + decoding:1
parser.add_argument('-p', '--prefill_length', type=int, required=False, default=100)  # default: length of init input


arguments = parser.parse_args()
server_ip = arguments.addr
rank = arguments.rank
world_size = arguments.world_size
repeat = arguments.repeat
split_embedding = bool(arguments.se)
generate_length = arguments.generate_length
prefill_length = arguments.prefill_length

# Enable when using RaspberryPi
if distributed:
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'


def log(log_type: str, message: str):
    print(f'[{get_timestamp()} {log_type}]: {message}')


class Worker:
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
        self.split_embedding = None

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

    def init_weights(self, data: tuple):
        if self.rank == 0:
            self.config, self.split_weights, self.tokenizer, embedding_weights, self.ln_f_weight = data

            token_embedding_weight, position_embedding_weight = embedding_weights
            self.token_embedding = nn.Embedding.from_pretrained(token_embedding_weight)
            self.position_embedding = nn.Embedding.from_pretrained(position_embedding_weight)
            self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(token_embedding_weight)

        else:  # other workers
            self.config, self.split_weights = data

    def init(self, split_MLP=True, split_embedding=False):
        """
        Receive split_weight and config from master
        """
        # file_name for the split_weights
        split_weights_dir = 'split_weights/'
        if not os.path.exists(split_weights_dir):
            os.makedirs(split_weights_dir)
            log('Notice', 'Directory of split_weights not found, create the directory')

        split_weights_file = f'{model_name}-n={self.n_device}-r={self.rank}-se={1 if split_embedding else 0}.sw'
        data = None  # split_weights
        if not os.path.exists(split_weights_dir + split_weights_file):
            log('Notice', f'File of "{split_weights_file}" not found, download from Master')

            with socket.create_connection((MASTER_IP, master_port)) as conn:
                if self.rank == 0:  # main worker
                    log('Notice', 'Send WORKER_NUM to master')
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
                self.split_embedding = split_embedding
                log('Notice', 'Get necessary weights from master')

                if split_embedding:
                    request = f'SE_WEIGHT {self.rank}\n'
                else:
                    request = f'TP_WEIGHT {self.rank} SPLIT_MLP {int(split_MLP)}\n'

                conn.sendall(request.encode())
                data, raw_data = recv_data(conn, return_bytes=True)

                with open(split_weights_dir + split_weights_file, 'wb') as f:
                    f.write(raw_data)
                del raw_data
        else:
            print(f'[{get_timestamp()} Notice]: File "{split_weights_file}" found, load from cache')
            with open(split_weights_dir + split_weights_file, 'rb') as f:
                data = pickle.load(f)

        self.init_weights(data)
        if split_embedding:
            h_d = self.config.h // self.n_device
            self.split_heads = self.rank * h_d, (self.rank + 1) * h_d  # default: equal split
            h_l, h_r = self.split_heads
            self.split_embedding_range = h_l * self.config.d_h, h_r * self.config.d_h  # default: equal split
        log('Notice', 'TP Init Done!')

    def init_inference(self, input_ids=None):
        # init cache
        if self.split_KV_cache is None:
            past_length = 0
            self.split_KV_cache = (None,) * self.config.n_layer
        else:
            past_length = self.split_KV_cache[0][0].size(-2)

        # init hidden_states
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
        return hidden_states

    def co_forward_tp(self, input_ids=None, use_cache=True, split_MLP=True):
        """
        :param input_ids: only the rank-0 device needs input
        :param use_cache: use KV-cache to reduce duplicated computation
        :return:
        """
        # split_heads = self.config.split_heads
        d_model = self.config.d_model

        if show_latency:
            # latency_measure
            layer_latency = []
            ln_latency = []
            qkv_latency = []
            attn_latency = []
            Wo_latency = []
            mlp1_latency = []
            mlp2_latency = []

            all_reduce_latency = []
        if use_cache:
            cache_present = ()

        # init inference: obtain hidden_states and broadcast across devices
        hidden_states = self.init_inference(input_ids)

        # forward_layers
        for layer_idx in range(self.config.n_layer):
            start_layer = time.perf_counter()

            split_weight_layer = self.split_weights[layer_idx]
            ln1_w_b, ln2_w_b = split_weight_layer['LN']
            split_cache_layer = self.split_KV_cache[layer_idx]

            residual = hidden_states

            start_ln = time.perf_counter()
            # LN1
            hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, self.config.ln_eps)
            if show_latency:
                end_ln = time.perf_counter()
                ln_latency.append(end_ln - start_ln)

            # split MHA
            attn_proj_w_b, attn_Wo_w_b = split_weight_layer['MHA']
            # hidden_states, layer_cache = MHA_forward_use_weights(hidden_states, split_MHA_weight, self.config,
            #                                                         split_cache_layer)

            start_qkv = time.perf_counter()
            # QKV projection
            QKV = Conv1D_forward_use_weights(hidden_states, attn_proj_w_b)
            if show_latency:
                end_qkv = time.perf_counter()
                qkv_latency.append(end_qkv - start_qkv)

            Q, K, V = split_QKV(QKV, self.config.d_h)

            if split_cache_layer is not None:
                # update layer_cache
                K_cache, V_cache = split_cache_layer
                K = torch.cat((K_cache, K), dim=2)
                V = torch.cat((V_cache, V), dim=2)
            split_cache_layer = (K, V)

            start_attn = time.perf_counter()
            attn_output, attn_weights = attn(Q, K, V)
            attn_output = merge_heads(attn_output, self.config.d_h)
            if show_latency:
                end_attn = time.perf_counter()
                attn_latency.append(end_attn - start_attn)

            start_Wo = time.perf_counter()
            # Wo projection
            hidden_states = Conv1D_forward_use_weights(attn_output, attn_Wo_w_b)
            if show_latency:
                end_Wo = time.perf_counter()
                Wo_latency.append(end_Wo - start_Wo)

            if use_cache:
                cache_present = cache_present + (split_cache_layer,)

            if split_MLP:  # split MLP inference on workers
                # AllReduce
                start_ar = time.perf_counter()
                # dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
                hidden_states = self.all_reduce(hidden_states)
                if show_latency:
                    end_ar = time.perf_counter()
                    all_reduce_latency.append(end_ar - start_ar)

                # Residual connection
                hidden_states = hidden_states + residual

                if self.rank == 0:  # central
                    residual = hidden_states

                start_ln = time.perf_counter()
                # LN2
                hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, self.config.ln_eps)
                if show_latency:
                    end_ln = time.perf_counter()
                    ln_latency.append(end_ln - start_ln)

                # split MLP
                mlp1_w_b, mlp2_w_b = split_weight_layer['MLP']
                # split_MLP_output = MLP_forward_use_weights(hidden_states, split_MLP_weight, self.config)

                start_mlp1 = time.perf_counter()
                # mlp1
                hidden_states = Conv1D_forward_use_weights(hidden_states, mlp1_w_b)
                if show_latency:
                    end_mlp1 = time.perf_counter()
                    mlp1_latency.append(end_mlp1 - start_mlp1)

                # activation
                hidden_states = self.MLP_activation(hidden_states)

                start_mlp2 = time.perf_counter()
                # mlp2
                hidden_states = Conv1D_forward_use_weights(hidden_states, mlp2_w_b)
                if show_latency:
                    end_mlp2 = time.perf_counter()
                    mlp2_latency.append(end_mlp2 - start_mlp2)

                # AllReduce or SingleReduce (for the last layer)
                start_ar = time.perf_counter()
                if layer_idx == self.config.n_layer - 1:
                    dist.reduce(hidden_states, dst=0, op=dist.ReduceOp.SUM)
                else:
                    # dist.all_reduce(split_MLP_output, op=dist.ReduceOp.SUM)
                    split_MLP_output = self.all_reduce(hidden_states)
                if show_latency:
                    end_ar = time.perf_counter()
                    all_reduce_latency.append(end_ar - start_ar)

                # Residual connection
                hidden_states = split_MLP_output + residual

            else:  # inference MLP only on device 0
                dist.reduce(hidden_states, dst=0, op=dist.ReduceOp.SUM)
                if self.rank == 0:
                    hidden_states = hidden_states + residual
                    residual = hidden_states
                    # LN2
                    hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, self.config.ln_eps)
                    # MLP
                    MLP_weight_layer = self.layers_MLP_weights[layer_idx]
                    hidden_states = MLP_forward_use_weights(hidden_states, MLP_weight_layer, self.config)
                    # Residual connection
                    hidden_states = hidden_states + residual
            if show_latency:
                end_layer = time.perf_counter()
                layer_latency.append(end_layer - start_layer)

        if use_cache:  # udpate cache
            self.split_KV_cache = cache_present

        if show_latency:
            print(
                f'Layer forward  : total {np.sum(layer_latency):.6f}s, avg {np.mean(layer_latency):.6f}s, max={np.argmax(layer_latency)} {np.max(layer_latency):.6f}s')
            print(
                f'LayerNorm      : total {np.sum(ln_latency):.6f}s, avg {np.mean(ln_latency):.6f}s, max={np.argmax(ln_latency)} {np.max(ln_latency):.6f}s')
            print(
                f'QKV projection : total {np.sum(qkv_latency):.6f}s, avg {np.mean(qkv_latency):.6f}s, max={np.argmax(qkv_latency)} {np.max(qkv_latency):.6f}s')
            print(
                f'Attention      : total {np.sum(attn_latency):.6f}s, avg {np.mean(attn_latency):.6f}s, max={np.argmax(attn_latency)} {np.max(attn_latency):.6f}s')
            print(
                f'Wo projection  : total {np.sum(Wo_latency):.6f}s, avg {np.mean(Wo_latency):.6f}s, max={np.argmax(Wo_latency)} {np.max(Wo_latency):.6f}s')
            print(
                f'MLP1 projection: total {np.sum(mlp1_latency):.6f}s, avg {np.mean(mlp1_latency):.6f}s, max={np.argmax(mlp1_latency)} {np.max(mlp1_latency):.6f}s')
            print(
                f'MLP2 projection: total {np.sum(mlp2_latency):.6f}s, avg {np.mean(mlp2_latency):.6f}s, max={np.argmax(mlp2_latency)} {np.max(mlp2_latency):.6f}s')
            print(f'All-Reduce: total {np.sum(all_reduce_latency)}s, avg {np.mean(all_reduce_latency)}s')

        if self.rank == 0:
            hidden_states = F.layer_norm(hidden_states, (d_model,), *self.ln_f_weight, self.config.ln_eps)
            return hidden_states

    def co_forward_se(self, input_ids=None, use_cache=True):
        """
        :param input_ids: only the rank-0 device needs input
        :param use_cache: use KV-cache to reduce duplicated computation
        :return:
        """
        # split_heads = self.config.split_heads
        d_model = self.config.d_model
        d_h = self.config.d_h

        if show_latency:
            layer_latency = []
            ln_latency = []
            qkv_latency = []
            attn_latency = []
            Wo_latency = []
            mlp1_latency = []
            mlp2_latency = []

        if use_cache:
            cache_present = ()

        # init inference: obtain hidden_states and broadcast across devices
        hidden_states = self.init_inference(input_ids)

        # forward layers
        for layer_idx, (layer_weights, layer_cache) in enumerate(zip(self.split_weights, self.split_KV_cache)):
            start_layer = time.perf_counter()
            # split_layer_norm_weights
            ln1_w_b, ln2_w_b = layer_weights['LN']

            # MHA{
            QKV_proj_w_b, Wo_proj_w_b = layer_weights['MHA']

            # merge QKV_proj_w_b: already merged in master
            # if len(QKV_proj_w_b) == 3:
            #     QKV_proj_w_b = tuple([torch.concat(ws_or_bs, dim=-1) for ws_or_bs in zip(*QKV_proj_w_b)])

            if layer_idx == 0:  # 1st layer
                # residual 1
                se_l, se_r = self.split_embedding_range
                residual = hidden_states[..., se_l:se_r]
                # LN1: complete weight
                hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, self.config.ln_eps)

                start_qkv_p = time.perf_counter()
                # QKV projection
                QKV = Conv1D_forward_use_weights(hidden_states, QKV_proj_w_b)

            else:  # 2-n_l layers
                # residual 1
                residual = split_embeddings

                start_ln = time.perf_counter()
                # LN1: split_embedding
                split_embeddings = layer_norm_se(split_embeddings, ln1_w_b, d_model, self.config.ln_eps)
                if show_latency:
                    end_ln = time.perf_counter()
                    ln_latency.append(end_ln - start_ln)

                start_qkv_p = time.perf_counter()
                # QKV projection: split_embedding
                QKV = self.ring_gather_reduce_comp_overlap(split_embeddings, *QKV_proj_w_b)
            if show_latency:
                end_qkv = time.perf_counter()
                qkv_latency.append(end_qkv - start_qkv_p)

            start_attn = time.perf_counter()
            Q, K, V = split_QKV(QKV, d_h)

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
            if show_latency:
                end_attn = time.perf_counter()
                attn_latency.append(end_attn - start_attn)

            start_Wo = time.perf_counter()
            # Wo projection
            split_embeddings = self.ring_reduce_scatter_comp_overlap(split_embeddings, *Wo_proj_w_b)
            if show_latency:
                end_Wo = time.perf_counter()
                Wo_latency.append(end_Wo - start_Wo)

            # residual connection 1
            split_embeddings = residual + split_embeddings
            # }MHA

            # residual 2
            residual = split_embeddings

            # MLP
            mlp1_w_b, mlp2_w_b = layer_weights['MLP']

            start_ln = time.perf_counter()
            # LN2: split_embedding
            split_embeddings = layer_norm_se(split_embeddings, ln2_w_b, d_model, self.config.ln_eps)
            end_ln = time.perf_counter()
            if show_latency:
                ln_latency.append(end_ln - start_ln)

            start_mlp1 = time.perf_counter()
            # MLP1
            split_embeddings = self.ring_gather_reduce_comp_overlap(split_embeddings, *mlp1_w_b)
            if show_latency:
                end_mlp1 = time.perf_counter()
                mlp1_latency.append(end_mlp1 - start_mlp1)

            # activation
            split_embeddings = self.MLP_activation(split_embeddings)

            start_mlp2 = time.perf_counter()
            if layer_idx != self.config.n_layer - 1:
                # MLP2
                split_embeddings = self.ring_reduce_scatter_comp_overlap(split_embeddings, *mlp2_w_b)

                # residual connection 2
                split_embeddings = residual + split_embeddings

            else:  # MLP2 of the final layer
                # MLP2
                hidden_states = Conv1D_forward_use_weights(split_embeddings, mlp2_w_b)

                # residual connection 2: partial
                hidden_states[..., se_l:se_r] += residual

                # reduce to rank0 device
                dist.reduce(hidden_states, dst=0)

                # residual_gather_task.wait()
            if show_latency:
                end_layer = time.perf_counter()
                mlp2_latency.append(end_layer - start_mlp2)
                layer_latency.append(end_layer - start_layer)

        if use_cache:  # udpate cache
            self.split_KV_cache = cache_present

        if show_latency:
            print(
                f'Layer forward  : total {np.sum(layer_latency):.6f}s, avg {np.mean(layer_latency):.6f}s, max={np.argmax(layer_latency)} {np.max(layer_latency):.6f}s')
            print(
                f'LayerNorm      : total {np.sum(ln_latency):.6f}s, avg {np.mean(ln_latency):.6f}s, max={np.argmax(ln_latency)} {np.max(ln_latency):.6f}s')
            print(
                f'QKV projection : total {np.sum(qkv_latency):.6f}s, avg {np.mean(qkv_latency):.6f}s, max={np.argmax(qkv_latency)} {np.max(qkv_latency):.6f}s')
            print(
                f'Attention      : total {np.sum(attn_latency):.6f}s, avg {np.mean(attn_latency):.6f}s, max={np.argmax(attn_latency)} {np.max(attn_latency):.6f}s')
            print(
                f'Wo projection  : total {np.sum(Wo_latency):.6f}s, avg {np.mean(Wo_latency):.6f}s, max={np.argmax(Wo_latency)} {np.max(Wo_latency):.6f}s')
            print(
                f'MLP1 projection: total {np.sum(mlp1_latency):.6f}s, avg {np.mean(mlp1_latency):.6f}s, max={np.argmax(mlp1_latency)} {np.max(mlp1_latency):.6f}s')
            print(
                f'MLP2 projection: total {np.sum(mlp2_latency):.6f}s, avg {np.mean(mlp2_latency):.6f}s, max={np.argmax(mlp2_latency)} {np.max(mlp2_latency):.6f}s')

        if self.rank == 0:
            # LN_f
            hidden_states = F.layer_norm(hidden_states, (d_model,), *self.ln_f_weight, self.config.ln_eps)
            return hidden_states

    def get_ring_index(self, index: int, offset: int):  # ring index starts from 0
        ring_index = index + offset
        if ring_index < 0:
            ring_index += self.n_device
        elif ring_index >= self.n_device:
            ring_index %= self.n_device
        return ring_index

    # call dist.d2d comm (send, recv) to implement all-reduce
    def reduce_scatter(self, xi: torch.Tensor, split_dim=-1):
        """
        :param split_dim: split_dim=-1 for TP
        :return:
        """
        assert xi.size(-1) % self.n_device == 0
        se = xi.size(-1) // self.n_device
        xis_p = [xi_p.contiguous() for xi_p in xi.split(se, dim=split_dim)]
        from_index = self.get_ring_index(self.rank, -1)
        from_tensor = torch.empty_like(xis_p[0])
        to_index = self.get_ring_index(self.rank, 1)
        # n-1 rounds comm
        for i in range(self.n_device - 1, 0, -1):
            send_idx = self.get_ring_index(self.rank, i)
            recv_idx = self.get_ring_index(send_idx, -1)

            recv_task = dist.irecv(from_tensor, src=from_index)
            send_task = dist.isend(xis_p[send_idx], dst=to_index)

            # send_task.wait()
            recv_task.wait()

            xis_p[recv_idx].add_(from_tensor)

        return xis_p[self.rank]

    def all_gather(self, x_p):
        to_gathers = [x_p if i == self.rank else torch.empty_like(x_p) for i in range(self.n_device)]
        from_index = self.get_ring_index(self.rank, -1)
        to_index = self.get_ring_index(self.rank, 1)
        send_idx = self.rank
        recv_idx = self.get_ring_index(send_idx, -1)

        for i in range(world_size - 1):
            recv_task = dist.irecv(to_gathers[recv_idx], src=from_index)
            send_task = dist.isend(to_gathers[send_idx], dst=to_index)

            recv_task.wait()
            send_idx = recv_idx
            recv_idx = self.get_ring_index(send_idx, -1)

        return to_gathers

    def all_reduce(self, xi, split_dim=-1):
        x_p = self.reduce_scatter(xi, split_dim)
        x_ps = self.all_gather(x_p)
        return torch.concat(x_ps, dim=split_dim)

    # todo: 这玩意为什么会有问题啊
    # def ring_reduce_scatter_comp_overlap(self, xi: torch.Tensor, Wi: torch.Tensor, bi=None):
    #     """
    #     overlap the ring reduce-scatter comm operation with the **previous** matrix-vector multiplication
    #     :return: split reduced results
    #     """
    #     W_in, W_out = Wi.shape
    #     # the whole weight matrix is split along row dimension(dim=0) across devices
    #     assert xi.size(-1) == W_in and W_out % self.n_device == 0
    #
    #     # further split partial weight along column dimension(dim=1)
    #     W_split_size = W_out // self.n_device
    #     Wi_split = Wi.split(W_split_size, dim=-1)
    #
    #     comm_from_index = self.get_ring_index(self.rank, -1)
    #     from_tensor = torch.zeros(*xi.shape[:-1], W_split_size)
    #     comm_to_index = self.get_ring_index(self.rank, 1)
    #
    #     comp_index = self.get_ring_index(self.rank, -1)
    #     yi = torch.matmul(xi, Wi_split[comp_index])
    #
    #     for i in range(0, self.n_device - 1):  # reverse order from n-1 to 0
    #         # start receiving
    #         send_task = dist.isend(yi, dst=comm_to_index)
    #         recv_task = dist.irecv(from_tensor, src=comm_from_index)
    #
    #         # partial computation
    #         comp_index = self.get_ring_index(comp_index, -1)  # [rank-1, rank-2, ..., rank]
    #         yi = torch.matmul(xi, Wi_split[comp_index])
    #
    #         # recv result from others
    #         # comment the send_task.wait() for distributed=True
    #         # if not distributed:
    #         #     send_task.wait()
    #         recv_task.wait()
    #         yi.add_(from_tensor)  # reduce
    #     if bi is not None:
    #         yi.add_(bi)
    #     return yi

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

        for i in range(self.n_device - 1, -1, -1):  # reverse order from n-1 to 0

            comp_index = self.get_ring_index(self.rank, i)  # [rank-1, rank-2, ..., rank]
            # start receiving
            if i != self.n_device - 1:
                send_task = dist.isend(yi, dst=comm_to_index)
                recv_task = dist.irecv(from_tensor, src=comm_from_index)

            else:
                recv_task = None

            # partial computation
            yi = torch.matmul(xi, Wi_split[comp_index])

            # recv result from others
            if recv_task:
                # comment the send_task.wait() for distributed=True
                if not distributed:
                    send_task.wait()
                recv_task.wait()
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

    # def co_forward_tsp(self, input_ids=None, use_cache=True):

    def tp_generate(self, input_len=100, generate_len=100, split_MLP=True, return_latency=False):
        # todo: modify the max_length to generate_length
        if split_embedding:
            tp_forward = self.co_forward_se
        else:
            tp_forward = self.co_forward_tp

        max_input_length = len(self.text)
        assert input_len > 0
        input_len = min(max_input_length, input_len)
        assert input_len + generate_len <= 1024
        log('Task', f'input_length={input_len}, generate_length={generate_len}')

        self.split_KV_cache = None  # clear cache

        with torch.no_grad():  # no_grad() to release memory
            # prefill
            if self.rank == 0:
                input_ids = torch.LongTensor([self.tokenizer.convert_tokens_to_ids(list(self.text[:input_len]))])
                if self.batch_size > 1:
                    input_ids = torch.concat([input_ids] * self.batch_size, dim=0)
                # cur_len = input_ids.size(-1)
                # assert cur_len < max_length
                # share the cur_len
                input_size = torch.tensor(input_ids.shape, dtype=torch.int32)
                dist.broadcast(input_size, src=0)

                generated_ids = []

                log('Task', 'Start Prefill')
                start_prefill = time.perf_counter()
                hidden_states = tp_forward(input_ids)

                start_lmhead = time.perf_counter()
                logits = self.lm_head(hidden_states[:, -1, :])
                next_token_ids = logits2token(logits)
                input_ids = next_token_ids

                end_prefill = time.perf_counter()
                prefill_t = end_prefill - start_prefill
                log('Task', f'Prefill : {prefill_t}s')

                generated_ids.append(next_token_ids)
                if show_latency:
                    log('Task', f'{end_prefill - start_lmhead}s for lm_head and token generation')

            else:  # other workers
                # input_size = torch.empty(2, dtype=torch.int32)
                # dist.broadcast(input_size, src=0)
                # cur_len = input_size[-1]
                log('Task', 'Start Prefill')
                tp_forward()

            # cur_len += 1

            # decoding
            log('Task', 'Start decoding')
            start_decoding = time.perf_counter()
            for seq_len in tqdm(range(1, generate_len)):
                if self.rank == 0:
                    # process = psutil.Process(os.getpid())
                    # log('Monitor', f'Current RSS: {process.memory_info().rss>>20:.2f} MiB')

                    hidden_states = tp_forward(input_ids)
                    logits = self.lm_head(hidden_states[:, -1, :])
                    next_token_ids = logits2token(logits)
                    generated_ids.append(next_token_ids)

                    input_ids = next_token_ids
                else:
                    tp_forward()

            end_decoding = time.perf_counter()

        if self.rank == 0:
            decoding_t = end_decoding - start_decoding
            log('Task', f'Decoding: {decoding_t}s')
            generated_text = self.tokenizer.convert_ids_to_tokens(generated_ids)
            output = (''.join(generated_text),)
            if return_latency:
                output = output + (prefill_t, decoding_t)
            return output


def layer_norm_se(split_embeddings, split_LN_weights, d_model, eps):
    x_mean = split_embeddings.sum(dim=-1, keepdim=True).div(d_model)
    # start_comm = time.perf_counter()
    # obtain x.SUM
    dist.all_reduce(x_mean, op=dist.ReduceOp.SUM)
    # x_mean.div_(d_model)
    split_embeddings = split_embeddings - x_mean  # attention: cannot modify the original split_embeddings
    x_var = split_embeddings.square().sum(dim=-1, keepdim=True).div(d_model)
    # obtain x.SSD (Sum of Squared Deviations)
    dist.all_reduce(x_var, op=dist.ReduceOp.SUM)
    # ssd_all_reduce_task.wait()
    # end_comm = time.perf_counter()
    x_var.add_(eps)

    # perform partial LayerNorm on split_embedding
    split_embeddings.div_(torch.sqrt(x_var))
    if split_LN_weights is not None:
        split_w, split_b = split_LN_weights
        return torch.addcmul(split_b, split_embeddings, split_w)  # split_embedding * split_w + split_b
    return split_embeddings


if __name__ == '__main__':
    worker = Worker((MAIN_WORKER_IP, port_tcp))
    show_latency = False
    batch = 1
    # split_embedding = False
    return_latency = True

    params = {
        'split_embedding': split_embedding,
    }
    worker.batch_size = batch
    worker.init(**params)

    print(f'- Model={model_tag}')
    print(f'- split_embedding={split_embedding}')
    print(f'- batch={batch}')
    print(f'- model_config:\n{worker.config}')
    print(worker.text)
    repeat_latency = []

    for r in range(repeat):
        output = worker.tp_generate(5, 2, return_latency=return_latency)

        if worker.rank == 0:
            if return_latency:
                generated_text, t_prefill, t_decoding = output
                repeat_latency.append([t_prefill, t_decoding])
            else:
                generated_text = output[0]
            print(generated_text)

    if worker.rank == 0 and return_latency:
        repeat_latency = np.asarray(repeat_latency)
        print(repeat_latency)
        print(repeat_latency.mean(axis=0))

    if dist.is_initialized():
        dist.destroy_process_group()
