import torch.distributed as dist
import torch
from network_config import *
import os
# from comm import send_data, recv_data
import time

# os.environ['GLOO_SOCKET_IFNAME'] = 'eth2'
# os.environ['GLOO_SOCKET_IFACE_NAME'] = 'eth2'  # 将 'eth0' 替换为你的网络接口名称


def main():
    # 初始化分布式环境：通过rank来区分设备
    print('init_method:', init_method)
    dist.init_process_group(backend='gloo', init_method=init_method, world_size=2, rank=1)

    print(dist.is_initialized())

    # 创建一个张量
    tensor = torch.randn(1, 64, 224, 224)
    print(f"Client sending tensor of {tensor.shape}")

    start = time.perf_counter()

    # 发送张量到服务器 (rank 1)
    dist.send(tensor, dst=0)

    # 接收从服务器返回的张量
    received_tensor = torch.zeros_like(tensor)
    dist.recv(received_tensor, src=0)
    end = time.perf_counter()
    assert torch.equal(tensor, received_tensor)
    print(f"Client received tensor of {received_tensor.shape}")

    consumption = end - start
    print(f'{consumption}s for transmission')


if __name__ == "__main__":
    main()
