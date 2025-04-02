import torch.distributed as dist
import torch
import threading
from queue import Queue



class CommHandler:
    """
    Handle communication between pipeline stages
    All devices:
    - recv from last rank
    - send to next rank
    - broadcast from the first stage and last stage
    First stage (additionally):
    - send to last stage
    Last stage (additionally):
    - recv from first stage
    """
    def __init__(self, rank, world_size, backend='gloo', enable_broadcast=False, device=None):
        self.rank = rank
        self.last_rank = world_size - 1 if rank == 0 else rank - 1
        self.world_size = world_size
        self.backend = backend
        # self.device = device if device is not None else torch.device('cpu')
        self.max_head_len = 5  # head = {n_bytes | shape}
        self.tensor_dtype = {  # n_bytes -> torch.dtype
            2: torch.float16,
            4: torch.float32,
            8: torch.float64,
        }
        self.stop_event = threading.Event()
        self.threads = []
        
    def setup_queue(self):
        # send queue
        self.send_queue = Queue()

        # recv queue
        self.recv_from_queue = {self.last_rank: Queue()}  # recv from last rank for each stage
        if self.rank == self.world_size - 1:  # last stage
            self.recv_from_queue[0] = Queue()  # recv from first stage for last stage (additionally)
        
        if self.enable_broadcast:
            # broadcast queue
            if self.rank == 0:
                self.broadcast_send_queue = Queue()
            if self.rank != 0:
                self.broadcast_recv_queue[0] = Queue()
            if self.rank != self.world_size - 1:
                self.broadcast_recv_queue[self.world_size - 1] = Queue()
            
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
        dist.init_process_group(backend=self.backend, init_method='tcp://localhost:12345', rank=self.rank, world_size=self.world_size)

    def sendto(self, data, dst_rank):
        self.send_queue.put((data, dst_rank))

    def keep_sending(self, send_queue):
        while True:
            data, dst = send_queue.get()
            head = self.get_head(data)
            dist.send(head, dst=dst)
            dist.send(data, dst=dst)
    
    def keep_receiving(self, src_rank):
        recv_queue = self.recv_from_queue[src_rank]
        while True:
            head = torch.zeros(self.max_head_len, dtype=torch.long)
            dist.recv(head, src=src_rank)
            dtype, tensor_shape = self.read_head(head)
            data = torch.zeros(tensor_shape, dtype=dtype)
            dist.recv(data, src=src_rank)
            recv_queue.put((src_rank, data))

    def recvfrom(self, src_rank):
        return self.recv_from_queue[src_rank].get()
    
    def start_thread(self, func, args):
        thread = threading.Thread(target=func, args=args, daemon=True)
        thread.start()
        self.threads.append(thread)

    def broadcast_send(self, data):
        self.broadcast_send_queue.put(data)

    def keep_broadcasting_send(self):
        while True:
            data = self.broadcast_send_queue.get()
            head = self.get_head(data)
            dist.broadcast(head, src=self.rank)
            dist.broadcast(data, src=self.rank)

    def keep_broadcasting_recv(self, src_rank):
        broadcast_recv_queue = self.broadcast_recv_queue[src_rank]
        while True:
            head = torch.zeros(self.max_head_len, dtype=torch.long)
            dist.broadcast(head, src=src_rank)
            dtype, tensor_shape = self.read_head(head)
            data = torch.zeros(tensor_shape, dtype=dtype)
            dist.broadcast(data, src=src_rank)
            broadcast_recv_queue.put((src_rank, data))

    def broadcast(self, data):
        self.broadcast_send_queue.put(data)

    def start_threads(self):
        # send thread
        self.start_thread(self.keep_sending, (self.send_queue,))
        # recv thread
        for src_rank in self.recv_from_queue.keys():
            self.start_thread(self.keep_receiving, (src_rank,))

        if self.enable_broadcast:
            # broadcast send thread
            if hasattr(self, 'broadcast_send_queue'):
                self.start_thread(self.keep_broadcasting_send, ())
            # broadcast recv thread
            if hasattr(self, 'broadcast_recv_queue'):
                for src_rank in self.broadcast_recv_queue.keys():
                    self.start_thread(self.keep_broadcasting_recv, (src_rank,))
            
        


