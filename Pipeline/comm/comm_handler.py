import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Tuple, Optional
from queue import Queue, Empty
from datetime import timedelta
import traceback


class CommHandler:
    """
    Handle communication between pipeline stages
    """
    def __init__(self, rank, world_size, backend='gloo', enable_broadcast=False, max_workers=4, device=None):
        self.rank = rank
        self.world_size = world_size
        self.last_rank = world_size - 1 if rank == 0 else rank - 1
        self.backend = backend
        self.enable_broadcast = enable_broadcast
        self.max_workers = max_workers

        # self.device = device if device is not None else torch.device('cpu')
        self.max_head_len = 5  # head = {n_bytes | shape}
        self.tensor_dtype = {  # n_bytes -> torch.dtype
            2: torch.float16,
            4: torch.float32,
            8: torch.long,
        }
        self.comm_device = torch.device('cpu')
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._threads = []
        self.setup_queue()

    def setup_queue(self):
        """
        All devices:
        - recv from last rank
        - send to next rank
        - broadcast_recv from the first stage and last stage
        First stage (additionally):
        - send to last stage
        Last stage (additionally):
        - recv from first stage
        """
        # send queue
        self.send_queue = Queue()

        # recv mark
        rank_mark = [False] * self.world_size
        rank_mark[self.last_rank] = True
        if self.rank == self.world_size - 1:
            rank_mark[0] = True  # first stage send tree_info to last stage (additionally)

        if self.enable_broadcast:
            if self.rank != 0:
                rank_mark[0] = True  # recv from first stage [only for broadcast]
            if self.rank != self.world_size - 1:
                rank_mark[self.world_size - 1] = True  # recv from last stage [only for broadcast]

        # recv queue
        self.recv_from_queue = {}
        for i in range(self.world_size):
            if rank_mark[i]:
                self.recv_from_queue[i] = Queue()
            
    def get_head(self, data):
        """
        Get the head of the data
        """
        head = torch.zeros(self.max_head_len, dtype=torch.long)
        head[0] = data.element_size()  # n_bytes
        head[1:1+data.dim()] = torch.tensor(data.shape)
        return head
    
    def read_head(self, head):
        """
        Read the head of the data
        """
        # n_bytes = head[0].item()
        dtype = self.tensor_dtype[head[0].item()]
        head_shape = head[1:]
        tensor_shape = head_shape[head_shape > 0].tolist()
        return dtype, tensor_shape

    def init_PG(self, init_method=None):
        if init_method is None:
            init_method = 'env://'
        print(f"Initializing process group with backend {self.backend} and rank {self.rank} and world size {self.world_size}")
        # dist.init_process_group(backend=self.backend, init_method='tcp://localhost:12345', rank=self.rank, world_size=self.world_size)
        dist.init_process_group(
            backend=self.backend,
            init_method=init_method,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=30)
        )
        print(f'Rank {self.rank} initialized')

    def send_tensor(self, data, dst_rank, tag=0):
        head = self.get_head(data)
        dist.send(head, dst=dst_rank, tag=tag)
        dist.send(data, dst=dst_rank, tag=tag)

    def recv_tensor(self, src_rank, tag=0):
        head = torch.zeros(self.max_head_len, dtype=torch.long)
        dist.recv(head, src=src_rank, tag=tag)
        dtype, tensor_shape = self.read_head(head)
        data = torch.zeros(tensor_shape, dtype=dtype)
        dist.recv(data, src=src_rank, tag=tag)
        return data

    def sendto(self, data, dst_rank):
        self.send_queue.put((data.to(self.comm_device), dst_rank))

    def keep_sending(self):
        while self._running:
            try:
                data, dst = self.send_queue.get(timeout=0.1)
                self.send_tensor(data, dst)
                # head = self.get_head(data)
                # dist.send(head, dst=dst)
                # dist.send(data, dst=dst)
            except Empty:
                continue
            except Exception as e:  # break when error occurs
                print(f"Send error from rank {self.rank}->{dst}:\n{type(e)}: {e}")
                break

    def keep_receiving(self, src_rank):
        recv_queue = self.recv_from_queue[src_rank]
        while self._running:
            try:
                data = self.recv_tensor(src_rank)
                # head = torch.zeros(self.max_head_len, dtype=torch.long)
                # dist.recv(head, src=src_rank)
                # dtype, tensor_shape = self.read_head(head)
                # data = torch.zeros(tensor_shape, dtype=dtype)
                # dist.recv(data, src=src_rank)
                recv_queue.put(data)

            except Exception as e:  # break when error occurs
                print(f"Recv error from rank {src_rank}->{self.rank}:\n{type(e)}: {e}")
                # traceback.print_exc()
                break

    def recvfrom(self, src_rank, device=None):
        # print(f"Rank {self.rank} receiving from rank {src_rank}")
        data = self.recv_from_queue[src_rank].get()
        if device is not None:
            data = data.to(device)
        return data
    
    # def start_thread(self, func, args=None):
    #     if args is None:
    #         thread = self.executor.submit(func)
    #     else:
    #         thread = self.executor.submit(func, *args)
    #     self._threads.append(thread)

    def multi_sendto(self, data):
        """
        - Implement broadcast with sendto()
        - Recv broadcast by recvfrom()
        - Tag=1: broadcast
        """
        if not self.enable_broadcast:
            raise ValueError("Broadcast is not enabled")
        try:
            for dst_rank in range(self.world_size):
                if dst_rank == self.rank:
                    continue
                # self.send_queue.put((data, dst_rank))
                # self.executor.submit(self.send_tensor, data, dst_rank, tag=1)
                head = self.get_head(data)
                self.executor.submit(dist.send, head, dst_rank, tag=1)
            for dst_rank in range(self.world_size):
                if dst_rank == self.rank:
                    continue
                self.executor.submit(dist.send, data, dst_rank, tag=1)
        except Exception as e:
            print(f"Multi send error in rank {self.rank}: {e}")
            raise e
        
    def recvfrom_async(self, src_rank, tag=0):
        # print(f"Rank {self.rank} recv_async from rank {src_rank}")
        return self.executor.submit(self.recv_tensor, src_rank, tag)

    def broadcast_send(self, data):
        """
        - Implement broadcast with sendto()
        - Recv broadcast by recvfrom()
        """
        if not self.enable_broadcast:
            raise ValueError("Broadcast is not enabled")
        try:
            data = data.to(self.comm_device)
            head = self.get_head(data)
            dist.broadcast(head, src=self.rank)
            dist.broadcast(data, src=self.rank)
            
        except Exception as e:
            print(f"Broadcast error in rank {self.rank}: {e}")
            raise e

    def broadcast_send_async(self, data):
        try:
            return self.executor.submit(self.broadcast_send, data)
        except Exception as e:
            print(f"Broadcast send async error in rank {self.rank}: {e}")
            raise e

    def broadcast_recv(self, src_rank, device=None):
        try:
            head = torch.zeros(self.max_head_len, dtype=torch.long)
            dist.broadcast(head, src=src_rank)
            dtype, tensor_shape = self.read_head(head)
            data = torch.zeros(tensor_shape, dtype=dtype)
            dist.broadcast(data, src=src_rank)
            if device is not None:
                data = data.to(device)
            return data
        except Exception as e:
            print(f"Broadcast error in rank {self.rank}: {e}")
            raise e
    
    def broadcast_recv_async(self, src_rank, device=None):
        return self.executor.submit(self.broadcast_recv, src_rank, device)

    def broadcast_tree_global(self, lens_split, tree_pos_ids, tree_mask):
        try:
            dist.broadcast(lens_split, src=self.rank)
            dist.broadcast(tree_pos_ids, src=self.rank)
            dist.broadcast(tree_mask, src=self.rank)
        except Exception as e:
            print(f"Broadcast tree global error in rank {self.rank}: {e}")
            raise e

    def broadcast_tree_global_async(self, lens_split, tree_pos_ids, tree_mask):
        return self.executor.submit(self.broadcast_tree_global, lens_split.cpu(), tree_pos_ids.cpu(), tree_mask.cpu())

    def broadcast_tree_global_recv(self, src_rank):
        lens_split = torch.zeros(self.world_size, dtype=torch.long)
        dist.broadcast(lens_split, src=src_rank)
        draft_len = torch.sum(lens_split).item()
        tree_pos_ids = torch.zeros(draft_len, dtype=torch.long)
        dist.broadcast(tree_pos_ids, src=src_rank)
        tree_mask = torch.zeros(1, 1, draft_len, draft_len, dtype=torch.float32)
        dist.broadcast(tree_mask, src=src_rank)
        return lens_split, tree_pos_ids, tree_mask

    def broadcast_tree_global_recv_async(self, src_rank):
        return self.executor.submit(self.broadcast_tree_global_recv, src_rank)

    def start_threads(self):
        if self._running:
            return
        self._running = True

        # send thread
        # self.start_thread(self.keep_sending, self.send_queue)
        self._threads.append(self.executor.submit(self.keep_sending))
        # recv thread
        for src_rank in self.recv_from_queue.keys():
            # self.start_thread(self.keep_receiving, src_rank)
            self._threads.append(self.executor.submit(self.keep_receiving, src_rank))

    def stop(self):
        self._running = False
        for future in self._threads:
            future.cancel()
        self.executor.shutdown(wait=True)
    

            
        


