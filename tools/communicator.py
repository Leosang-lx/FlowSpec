# from loguru import logger

import os
import pickle

import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor

# def init_communication_group() -> None:
#     '''
#     Initialize the communication group for the distributed training.
#     '''
#     logger.info("Initializing Communication Group...")
#     dist.init_process_group(backend="gloo")
#     logger.info("Initialized Done!")
#     logger.info(
#         f"Rank: {dist.get_rank()} | World Size: {dist.get_world_size()}")

# def destroy_communication_group() -> None:
#     '''
#     Destroy the communication group.
#     '''
#     logger.info("Destroying Communication Group...")
#     dist.destroy_process_group()
#     logger.info("Destroyed Done!")


def send(data, dst: int) -> None:
    '''
    Send data to the destination process.
    data: the data to be sent.
    dst: the destination process.
    tag: the tag of the message.
    '''
    data_size = torch.tensor(list(data.shape),
                             dtype=torch.int64).to(torch.device('cpu'))
    # print(f"data_size: {data_size}")
    dist.send(data_size, dst=dst)
    data_tensor = data.to(torch.device('cpu'))
    # print(f"data: {data_tensor}")
    dist.send(data_tensor, dst=dst)


def recv(src: int, data_type, shape_length) -> object:
    '''
    Receive data from the source process.
    src: the source process.
    tag: the tag of the message.
    '''
    data_size = torch.zeros(shape_length, dtype=torch.int64).to(torch.device('cpu'))
    dist.recv(data_size, src=src)
    # print(f"data_size: {data_size}")
    if shape_length == 0:
        data_tensor = torch.empty((), dtype=data_type).to(torch.device('cpu'))
    else:
        data_tensor = torch.empty(*data_size.tolist(),
                              dtype=data_type).to(torch.device('cpu'))
    # print(f"data_tensor.shape: {data_tensor.shape}")
    dist.recv(data_tensor, src=src)
    # print(f"data_tensor: {data_tensor}")
    return data_tensor

def broadcast(data = None, src: int = 0, tag=0, shape_length=0, data_type = None) -> None:
    if data is not None:
        data_size = torch.tensor(list(data.shape),
                             dtype=torch.int64).to(torch.device('cpu'))
        dist.broadcast(data_size, src=src)
        data_tensor = data.to(torch.device('cpu'))
        dist.broadcast(data_tensor, src=src)
    else:
        data_size = torch.zeros(shape_length, dtype=torch.int64).to(torch.device('cpu'))
        dist.broadcast(data_size, src=src)
        if shape_length == 0:
            data_tensor = torch.empty((), dtype=data_type).to(torch.device('cpu'))
        else:
            data_tensor = torch.empty(*data_size.tolist(),
                              dtype=data_type).to(torch.device('cpu'))
        dist.broadcast(data_tensor, src=src)
        return data_tensor
    
    
# def broadcast(data, dsts: list, tag=0, enable_async=True) -> None:
#     '''
#     Broadcast data to all the destination processes.
#     data: the data to be broadcasted.
#     dsts: A list of destination processes.
#     tag: the tag of the message.
#     enable_async: whether to use async communication.
#     '''
#     if enable_async:
#         with ThreadPoolExecutor() as executor:
#             for dst in dsts:
#                 executor.submit(send, data, dst, tag)
#     else:
#         for dst in dsts:
#             send(data, dst, tag)