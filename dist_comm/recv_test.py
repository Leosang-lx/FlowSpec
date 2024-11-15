import torch
import torch.distributed as dist
import socket
# from comm import list_network_interfaces
import os
from network_config import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', required=False, default=None)

arguments = parser.parse_args()
provided_ip = arguments.server

init_method = gen_init_method(MAIN_WORKER_IP, port_torch)
print('init_method:', init_method)
# store = dist.TCPStore(server_ip, port, )
dist.init_process_group(backend='gloo', init_method=init_method, world_size=2, rank=0)
print(dist.is_initialized())

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


# os.environ['USE_LIBUV'] = 'False'
# os.environ['USE_IPv6'] = 'False'
# os.environ['MASTER_ADDR'] = server_ip
# os.environ['MASTER_PORT'] = str(port)
# Enable when using RaspberryPi
# os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--ip', required=False)
tensor_shape = (1, 64, 224, 224)


def recv_and_send():
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.bind((server_ip, port))
    # sock.listen()

    # 初始化分布式环境

    print(dist.get_rank())

    # 创建一个空的张量来接收客户端发送的数据
    received_tensor = torch.zeros(tensor_shape)  # 假设我们知道张量的形状

    # 从客户端 (rank 0) 接收张量
    dist.recv(received_tensor, src=1)
    print(f"Server received tensor: {received_tensor.shape}")

    # 立即发送接收到的张量回客户端
    dist.send(received_tensor, dst=1)

def test_broadcast():
    data = torch.randn(1, 100, 768)
    shape = torch.tensor(data.shape, dtype=torch.int32)
    print(shape.dtype)
    dist.broadcast(shape, src=0)
    dist.broadcast(data, src=0)
    print(data)


if __name__ == "__main__":
    test_broadcast()
