import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Tuple, Optional
from queue import Queue, Empty
from datetime import timedelta


class CommHandler:
    """
    Handle communication between pipeline stages
    """
    def __init__(self, rank, world_size, backend='gloo', enable_broadcast=False, max_workers=6, device=None):
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
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._threads = []
        
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

        # recv queue
        self.recv_from_queue = {self.last_rank: Queue()}  # recv from last rank for each stage
        if self.rank == self.world_size - 1:  # last stage
            self.recv_from_queue[0] = Queue()  # recv from first stage for last stage (additionally)
        
        # if self.enable_broadcast:
        #     # broadcast send queue
        #     if self.rank == 0 or self.rank == self.world_size - 1:
        #         self.broadcast_send_queue = Queue()

        #     # broadcast recv queue
        #     self.broadcast_recv_queue = {}
        #     if self.rank != 0:
        #         self.broadcast_recv_queue[0] = Queue()
        #     if self.rank != self.world_size - 1:
        #         self.broadcast_recv_queue[self.world_size - 1] = Queue()
            
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

    def init_process_group(self):
        print(f"Initializing process group with backend {self.backend} and rank {self.rank} and world size {self.world_size}")
        # dist.init_process_group(backend=self.backend, init_method='tcp://localhost:12345', rank=self.rank, world_size=self.world_size)
        dist.init_process_group(backend=self.backend, init_method='env://', rank=self.rank, world_size=self.world_size)
        print(f'Rank {self.rank} initialized')

    def sendto(self, data, dst_rank):
        self.send_queue.put((data, dst_rank))

    def keep_sending(self, send_queue):
        while self._running:
            try:
                data, dst = send_queue.get(timeout=0.1)
                head = self.get_head(data)
                dist.send(head, dst=dst)
                dist.send(data, dst=dst)
            except Empty:
                continue
            except Exception as e:  # break when error occurs
                print(f"Send error from rank {self.rank}->{dst}:\n{type(e)}: {e}")
                break

    def keep_receiving(self, src_rank):
        recv_queue = self.recv_from_queue[src_rank]
        while self._running:
            try:
                head = torch.zeros(self.max_head_len, dtype=torch.long)
                dist.recv(head, src=src_rank)
                dtype, tensor_shape = self.read_head(head)
                data = torch.zeros(tensor_shape, dtype=dtype)
                dist.recv(data, src=src_rank)
                recv_queue.put(data)

            except Exception as e:  # break when error occurs
                print(f"Recv error from rank {src_rank}->{self.rank}:\n{type(e)}: {e}")
                break

    def recvfrom(self, src_rank):
        # print(f"Rank {self.rank} receiving from rank {src_rank}")
        return self.recv_from_queue[src_rank].get()
    
    # def start_thread(self, func, args=None):
    #     if args is None:
    #         thread = self.executor.submit(func)
    #     else:
    #         thread = self.executor.submit(func, *args)
    #     self._threads.append(thread)

    # def broadcast_send(self, data):
    #     self.broadcast_send_queue.put(data)

    # def keep_broadcasting_send(self):
    #     while self._running:
    #         try:
    #             data = self.broadcast_send_queue.get()
    #             print(f"====Rank {self.rank} get broadcast data to send...")
    #             head = self.get_head(data)
    #             print(f'====Rank {self.rank} send broadcast head={head}...')
    #             dist.broadcast(head, src=self.rank)
    #             print(f'====Rank {self.rank} send broadcast head... done')
    #             print(f'====Rank {self.rank} send broadcast data...')
    #             dist.broadcast(data, src=self.rank)
    #             print(f'====Rank {self.rank} send broadcast data... done')
    #         except Exception as e:
    #             print(f"Broadcast send error from rank {self.rank}:\n{type(e)}: {e}")
    #             break

    # def keep_broadcasting_recv(self, src_rank):
    #     broadcast_recv_queue = self.broadcast_recv_queue[src_rank]
    #     while self._running:
    #         try:
    #             head = torch.zeros(self.max_head_len, dtype=torch.long)
    #             print(f"====Rank {self.rank} recv broadcast head from rank {src_rank}...")
    #             dist.broadcast(head, src=src_rank)
    #             print(f"====Rank {self.rank} recv broadcast head={head} from rank {src_rank}... done")
    #             dtype, tensor_shape = self.read_head(head)
    #             data = torch.zeros(tensor_shape, dtype=dtype)
    #             print(f"====Rank {self.rank} recv broadcast data from rank {src_rank}...")
    #             dist.broadcast(data, src=src_rank)
    #             print(f"====Rank {self.rank} recv broadcast data from rank {src_rank}... done")
    #             broadcast_recv_queue.put(data)
    #         except Exception as e:
    #             print(f"Broadcast recv error from rank {self.rank}:\n{type(e)}: {e}")
    #             break

    # def broadcast_recv(self, src_rank):
    #     return self.broadcast_recv_queue[src_rank].get()
    

    def start_threads(self):
        if self._running:
            return
        self._running = True

        # send thread
        # self.start_thread(self.keep_sending, self.send_queue)
        self._threads.append(self.executor.submit(self.keep_sending, self.send_queue))
        # recv thread
        for src_rank in self.recv_from_queue.keys():
            # self.start_thread(self.keep_receiving, src_rank)
            self._threads.append(self.executor.submit(self.keep_receiving, src_rank))

        # if self.enable_broadcast:
        #     # broadcast send thread
        #     if hasattr(self, 'broadcast_send_queue'):
        #         # self.start_thread(self.keep_broadcasting_send, ())
        #         self._threads.append(self.executor.submit(self.keep_broadcasting_send))
        #     # broadcast recv thread
        #     if hasattr(self, 'broadcast_recv_queue'):
        #         for src_rank in self.broadcast_recv_queue.keys():
        #             # self.start_thread(self.keep_broadcasting_recv, (src_rank,))
        #             self._threads.append(self.executor.submit(self.keep_broadcasting_recv, src_rank))

    def stop(self):
        self._running = False
        for future in self._threads:
            future.cancel()
        self.executor.shutdown(wait=True)
    

            
        


