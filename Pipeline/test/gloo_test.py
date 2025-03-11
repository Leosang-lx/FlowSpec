import torch
import torch.distributed as dist
import time 
from datetime import timedelta
import os
from tools.communicator import send, recv


def gloo_test():
    dist.init_process_group(
        backend='gloo',
        # init_method='env://',
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    if rank == 0:
        time.sleep(5)
        send_tensor = torch.zeros((1, 2, 4), dtype=torch.float16)
        send(send_tensor, dst=1)
    else:
        print(f"Rank {rank} waiting for data")
        data = recv(src=rank-1, data_type=torch.float16, shape_length=3)
        print(f"Rank {rank} received data: {data}")
        if rank < 3:
            send(data, dst=rank+1)
        else:
            print(f"Rank {rank} is the last rank")
            try:
                # time.sleep(10)
                send(data, dst=1)
            except Exception as e:
                print(f"the problem is here rank {rank}")
                print(f"Rank {rank} error: {e}")
    
    if rank == 1:
        # time.sleep(10)
        print(f"Rank {rank} is waiting for data")
        try:
            recv_tensor = recv(src=3, data_type=torch.float16, shape_length=3)
            print(f"Rank {rank} received data: {recv_tensor}")
            print("Done")
        except Exception as e:
            print(f"the problem is here rank {rank}")
            print("Done")
    
    print(f"Rank {rank} is waiting for barrier")
    dist.barrier()
    dist.destroy_process_group()