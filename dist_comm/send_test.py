import socket

import torch.distributed as dist
import torch
from dist_comm.network_config import *
from comm import send_data, recv_data
import time

# os.environ['USE_LIBUV'] = 'False'
# os.environ['GLOO_SOCKET_IFNAME'] = 'eth2'
# os.environ['GLOO_SOCKET_IFACE_NAME'] = 'eth2'  # 将 'eth0' 替换为你的网络接口名称

def socket_send_and_recv():
    server_socket = socket.create_server((MAIN_WORKER_IP, port_tcp), family=ipvx)
    end_conn, addr = server_socket.accept()

    tensor = torch.randn(1, 64, 224, 224)

    start = time.perf_counter()

    send_data(end_conn, tensor)

    data = recv_data(end_conn)

    end = time.perf_counter()

    consumption = end - start
    print(f'{consumption}s for transmission')

    end_conn.close()
    server_socket.close()



def send_and_recv():
    init_method = gen_init_method(MAIN_WORKER_IP, port_torch)
    # 初始化分布式环境：通过rank来区分设备
    print('init_method:', init_method)
    dist.init_process_group(backend='gloo', init_method=init_method, world_size=2, rank=0)

    print(dist.is_initialized())

    # 创建一个张量
    tensor = torch.randn(1, 64, 224, 224)
    print(f"Client sending tensor of {tensor.shape} of type {tensor.dtype}")

    tensor_split = tensor.split(112, dim=-1)[0]

    start = time.perf_counter()

    # 发送张量到服务器 (rank 1)
    dist.send(tensor_split, dst=1)

    # 接收从服务器返回的张量
    received_tensor = torch.zeros_like(tensor)
    dist.recv(received_tensor, src=1)
    end = time.perf_counter()
    assert torch.equal(tensor, received_tensor)
    print(f"Client received tensor of {received_tensor.shape}")

    consumption = end - start
    print(f'{consumption}s for transmission')
    dist.destroy_process_group()

def test_broadcast():
    shape = torch.zeros(3, dtype=torch.int32)
    print(shape.dtype)
    dist.broadcast(shape, src=1)
    data = torch.zeros(*shape)
    dist.broadcast(data, src=1)
    print(data)


if __name__ == "__main__":
    socket_send_and_recv()
    send_and_recv()
