import time

import torch
import torch.distributed as dist

from DistributedTP.worker import layer_norm_se
from DistributedTP.comm import send_data, recv_data
from dist_comm.network_config import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', required=False, default=None)

arguments = parser.parse_args()
provided_ip = arguments.server


def get_ip_address():
    # windows!!!
    hostname = socket.gethostname()
    ip_addresses = socket.gethostbyname_ex(hostname)[2]
    # 通常会返回多个 IP 地址（例如，本地回环地址 127.0.0.1 和实际的网络接口地址）
    # 我们可以过滤掉本地回环地址，只保留实际的网络接口地址
    ip_addresses = [ip for ip in ip_addresses if not ip.startswith("127.") and not ip.startswith("::1")]

    if ip_addresses:
        return ip_addresses  # 返回第一个非本地回环地址
    else:
        return None


# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--ip', required=False)
tensor_shape = (1, 64, 224, 112)


def dist_init(rank: int):
    init_method = gen_init_method(MAIN_WORKER_IP, port_torch)
    print('init_method:', init_method)
    # store = dist.TCPStore(server_ip, port, )
    dist.init_process_group(backend='gloo', init_method=init_method, world_size=2, rank=rank)
    print(dist.is_initialized())


def socket_recv_and_send():
    end_conn = socket.create_connection((MAIN_WORKER_IP, port_tcp))

    data = recv_data(end_conn)

    send_data(end_conn, data)

    end_conn.close()


def recv_and_send():
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.bind((server_ip, port))
    # sock.listen()
    # 初始化分布式环境
    print(dist.get_rank())

    # 创建一个空的张量来接收客户端发送的数据
    received_tensor = torch.zeros(tensor_shape)  # 假设我们知道张量的形状

    # 从客户端 (rank 0) 接收张量
    dist.recv(received_tensor, src=0)
    print(f"Server received tensor: {received_tensor.shape}")

    # 立即发送接收到的张量回客户端
    dist.send(received_tensor, dst=0)

    dist.destroy_process_group()


def test_broadcast():
    data = torch.randn(1, 100, 768)
    shape = torch.tensor(data.shape, dtype=torch.int32)
    print(shape.dtype)
    dist.broadcast(shape, src=1)
    dist.broadcast(data, src=1)
    print(data)

def test_distributed_layerNorm():
    layer_norm = layer_norm_se
    # layer_norm = sync_layer_norm
    split_embeddings = torch.randn(640)
    split_w = torch.randn(640)
    split_b = torch.randn(640)
    total_t = []
    comm_t = []
    for i in range(100):
        start_ln = time.perf_counter()
        y, t_comm = layer_norm(split_embeddings, (split_w, split_b), 1280, 1e-3)
        end_ln = time.perf_counter()
        total = end_ln - start_ln
        total_t.append(total)
        comm_t.append(t_comm)
    print('2 * All-Reduce')
    print(f'LayerNorm: Total:{sum(total_t):.6f}s, avg:{sum(total_t) / 100:.6f}s')
    print(f'Comm     : Total:{sum(comm_t):.6f}s, avg:{sum(comm_t) / 100:.6f}s')


if __name__ == "__main__":
    # socket_recv_and_send()
    dist_init(1)
    # recv_and_send()
    test_distributed_layerNorm()

