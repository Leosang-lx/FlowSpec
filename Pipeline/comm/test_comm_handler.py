import torch
import torch.distributed as dist
import time
import argparse
from comm_handler import CommHandler
import tqdm

def init_comm_handler(rank, world_size, backend='gloo', enable_broadcast=False):
    # Initialize communication handler
    comm = CommHandler(rank, world_size, backend=backend, enable_broadcast=enable_broadcast)
    comm.init_process_group()
    comm.setup_queue()
    print(f"Rank {rank} setup queue: {comm.recv_from_queue.keys()}")
    comm.start_threads()
    return comm


def stop_comm_handler(comm):
    print(f"Rank {comm.rank} stopping")
    comm.stop()
    dist.destroy_process_group()


def test_send_recv(comm, rank, world_size):
    # comm = init_comm_handler(rank, world_size, backend='gloo')

    # Test data
    start = time.perf_counter()
    for i in range(50):
        print(f"Rank {rank} i={i}")

        data_size = 50000 + i
        ones = torch.ones(data_size)
        test_tensor = ones * rank

        # try:
        if rank == world_size - 1:
            # First rank sends to last rank
            # print(f"Rank {rank} sending tensor to 0: {test_tensor}")
            comm.sendto(test_tensor, 0)
            time.sleep(1)  # Wait for transmission

        elif rank == 0:
            # Last rank receives from first rank
            time.sleep(1)
            recv_data = comm.recvfrom(world_size - 1)
            # print(f"Rank {rank} received tensor from {world_size - 1}: {recv_data}")
            assert torch.allclose(recv_data, ones * (world_size - 1))

        # Test communication between adjacent ranks
        if rank < world_size - 1:
            # print(f"Rank {rank} sending tensor to {rank + 1}: {test_tensor}")
            comm.sendto(test_tensor, rank + 1)
            time.sleep(1)

        if rank > 0:
            time.sleep(1)
            recv_data = comm.recvfrom(rank - 1)
            # print(f"Rank {rank} received tensor from {rank - 1}: {recv_data}")
            assert torch.allclose(recv_data, ones * (rank - 1))
    end = time.perf_counter()
    print(f"Rank {rank} time: {end - start}s")


# def test_broadcast(comm, rank, world_size):
#     # Initialize communication handler with broadcast enabled
#     # comm = CommHandler(rank, world_size, backend='gloo', enable_broadcast=True)
#     # comm.init_process_group()
#     # comm.setup_queue()
#     # comm.start_threads()

#     # Test data
#     test_tensor = torch.tensor([1.0, 2.0, 3.0]) * rank

#     # try:
#     if rank == 0:
#         # First rank broadcasts
#         print(f"Rank {rank} broadcasting tensor: {test_tensor}")
#         comm.broadcast_send(test_tensor)
#         time.sleep(1)
#     else:
#         # Other ranks receive broadcast
#         recv_data = comm.broadcast_recv(0)
#         print(f"Rank {rank} received broadcast from {0}: {recv_data}")
#         assert torch.allclose(recv_data, torch.tensor([1.0, 2.0, 3.0]) * 0)
    
#     dist.barrier()

#     if rank == world_size - 1:
#         # Last rank broadcasts
#         print(f"Rank {rank} broadcasting tensor: {test_tensor}")
#         comm.broadcast_send(test_tensor)
#         time.sleep(1)
#     else:
#         # Other ranks receive broadcast
#         recv_data = comm.broadcast_recv(world_size - 1)
#         print(f"Rank {rank} received broadcast from {world_size - 1}: {recv_data}")
#         assert torch.allclose(recv_data, torch.tensor([1.0, 2.0, 3.0]) * (world_size - 1))


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--test', type=str, choices=['send_recv', 'broadcast'], default='send_recv')
    # args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])


    comm = init_comm_handler(rank, world_size, backend='gloo')

    # if args.test == 'send_recv':
    try:
        test_send_recv(comm, rank, world_size)
        dist.barrier()
        print(f'Rank {rank} finished test_send_recv()... done')

    except Exception as e:
        print(f"Rank {rank} error: {e}")
    finally:
        stop_comm_handler(comm)

if __name__ == '__main__':
    import os
    main() 