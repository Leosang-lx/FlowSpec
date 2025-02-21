import torch.distributed as dist
import torch
from dist_comm.network_config import *
from DistributedTP.comm import send_data, recv_data
import time
from DistributedTP.worker import layer_norm_se


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


def dist_init(rank: int):
    init_method = gen_init_method(MAIN_WORKER_IP, port_torch)
    print('init_method:', init_method)
    # store = dist.TCPStore(server_ip, port, )
    dist.init_process_group(backend='gloo', init_method=init_method, world_size=2, rank=rank)
    print(dist.is_initialized())


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
    # socket_send_and_recv()
    dist_init(0)
    # send_and_recv()
    test_distributed_layerNorm()
