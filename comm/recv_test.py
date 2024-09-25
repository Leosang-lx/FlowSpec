import torch
import torch.distributed as dist
import socket
# from comm import list_network_interfaces
import os
from network_config import *
import argparse


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


# server_ip = get_ip_address()
# server_ip = '127.0.0.1'
# server_ip = '::1'
# os.environ['GLOO_SOCKET_IFNAME'] = 'eth2'
# os.environ['GLOO_SOCKET_IFACE_NAME'] = 'eth2'  # 将 'eth0' 替换为你的网络接口名称

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ip', required=False)

# init_method = f'tcp://{server_ip}:{port}'

tensor_shape = (1, 64, 224, 224)


def main():
    # 初始化分布式环境
    print('init_method:', init_method)
    dist.init_process_group(backend='gloo', init_method=init_method, world_size=2, rank=server_rank)
    print(dist.is_initialized())
    print(dist.get_rank())

    # 创建一个空的张量来接收客户端发送的数据
    received_tensor = torch.zeros(tensor_shape)  # 假设我们知道张量的形状

    # 从客户端 (rank 0) 接收张量
    dist.recv(received_tensor, src=1)
    print(f"Server received tensor: {received_tensor.shape}")

    # 立即发送接收到的张量回客户端
    dist.send(received_tensor, dst=1)


if __name__ == "__main__":
    main()
