import torch
import torch.distributed as dist
import time 
from datetime import timedelta
import os
from Pipeline.tools.communicator import send, recv
# def run():
#     # 初始化进程组，使用 Gloo 作为后端
#     dist.init_process_group(
#         backend='gloo',
#         # init_method='env://',
#     )
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()

#     # 设置设备（如果使用 GPU）
#     device = rank % torch.cuda.device_count()
#     torch.cuda.set_device(device)

#     # print(f"Rank {rank} ' ip {os.environ.get('MASTER_ADDR', '127.0.0.1')} ' port {os.environ.get('MASTER_PORT', '29500')}")
    
#     if rank == 0:
#         time.sleep(5)
#         send_tensor = torch.zeros((1, 2, 4), dtype=torch.float16)
#         send(send_tensor, dst=1)
#     else:
#         print(f"Rank {rank} waiting for data")
#         data = recv(src=rank-1, data_type=torch.float16, shape_length=3)
#         print(f"Rank {rank} received data: {data}")
#         if rank < 3:
#             send(data, dst=rank+1)
#         else:
#             print(f"Rank {rank} is the last rank")
#             try:
#                 # time.sleep(10)
#                 send(data, dst=1)
#             except Exception as e:
#                 print(f"the problem is here rank {rank}")
#                 print(f"Rank {rank} error: {e}")
    
#     if rank == 1:
#         # time.sleep(10)
#         print(f"Rank {rank} is waiting for data")
#         try:
#             recv_tensor = recv(src=3, data_type=torch.float16, shape_length=3)
#             print(f"Rank {rank} received data: {recv_tensor}")
#             print("Done")
#         except Exception as e:
#             print(f"the problem is here rank {rank}")
#             print("Done")
    
#     print(f"Rank {rank} is waiting for barrier")
#     dist.barrier()
#     dist.destroy_process_group()

def run():

    dist.init_process_group(backend='gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # print(f'rank={rank}, world_size={world_size}, device={device}')

    print(f"Rank {rank} ' ip {os.environ['MASTER_ADDR']} ' port {os.environ['MASTER_PORT']}")
    if rank == 0:
        time.sleep(5)
        send_tensor = torch.zeros((1,9,4096), dtype=torch.float16)
        # print(f"Rank 0 sending tensor: {send_tensor}")
        # dist.send(send_tensor, dst=1)
        send(send_tensor, dst=1)
    else:
        # data = torch.zeros(3).to("cpu")
        print(f"Rank {rank} waiting for data")
        # dist.recv(data, src=0)
        data = recv(src=rank-1, data_type=torch.float16, shape_length=3)
        print(f"Rank {rank} received data: {data}")
        if rank < 3:
            time.sleep(5)
            send(data, dst=rank+1)
        else:
            print(f"Rank {rank} is the last rank")
            try:
                
                time.sleep(5)
                send(data, dst=0)
            except Exception as e:
                print(f"the problem is here rank {rank}")
                # print(f"Rank {rank} error: {e}")
            

    if rank == 0:
        time.sleep(5)
        print(f"Rank {rank} is waiting for data")
        try:
            recv_tensor = recv(src=3, data_type=torch.float16, shape_length=3)
            print(f"Rank {rank} received data: {recv_tensor}")
            print("Done")
        except Exception as e:
            # print(f"Rank {rank} error: {e}")
            print(f"the problem is here rank {rank}")
            print("Done")
    
    print(f"Rank {rank} is waiting for barrier")
    dist.barrier()
    dist.destroy_process_group()    
    # tensor = torch.tensor([1, 2, 3], dtype=torch.int64).to("cpu")
    # tensor_size = torch.tensor(list(tensor.shape), dtype=torch.int64)
    # print(tensor_size)
    # print(*tensor.tolist())
    # tensor_new = torch.zeros(*tensor.tolist(), dtype=torch.int64)
    # print(tensor_new)

if __name__ == "__main__":
    run()
# import torch
# import torch.distributed as dist
# import os
# def run():
#     # 初始化进程组，使用 Gloo 作为后端
#     dist.init_process_group(
#         backend='gloo',
#         # init_method='env://',
#     )
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     # 创建一个张量
#     tensor = torch.tensor([rank], dtype=torch.float32).to("cpu")

#     # 如果是进程 0，发送张量到进程 1
#     if rank == 0:
#         # 发送到进程 1
#         dist.send(tensor=tensor, dst=1)
#         print(f"Rank {rank} sent data: {tensor.item()}")

#     # 如果是进程 1，接收张量从进程 0
#     elif rank == 1:
#         # 接收来自进程 0
#         dist.recv(tensor=tensor, src=0)
#         print(f"Rank {rank} received data: {tensor.item()}")

#     # 清理分布式环境
#     dist.destroy_process_group()

# if __name__ == "__main__":
#     # 获取当前进程的 rank 和世界大小
#     rank = int(os.environ['RANK'])
#     world_size = int(os.environ['WORLD_SIZE'])

#     # 启动运行函数
#     run()