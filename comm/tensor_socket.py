import zmq
import torch
import numpy as np
import struct
import socket
from subprocess import getstatusoutput
import threading
from queue import Queue, Empty
import time
import traceback
import sys

DTYPE_MAP = {
    1: torch.int8,
    2: torch.float16,
    3: torch.int32,
    4: torch.int64,
}

NP_DTYPE_MAP = {
    1: np.int8,
    2: np.float16,
    3: np.int32,
    4: np.int64,
}

MAX_DIMS = 4
HEADER_FORMAT = "6i"

PORT = 12345
SUBNET = '192.168.1'

def get_ip_addr(__subnet__: str):  # get ip by the prefix of __subnet__
    subnet = __subnet__ if __subnet__ else SUBNET
    status, ip_addr = getstatusoutput(f'ifconfig | grep "{subnet}" | awk \'{{print $2}}\'')
    if status == 0:
        return ip_addr
    return None

def get_all_ips():
    hostname = socket.gethostname()
    try:
        ip_list = socket.getaddrinfo(hostname, None)
        ips = set()
        for result in ip_list:
            ips.add(result[4][0])
        return list(ips)
    except:
        return []

def gen_header(tensor: torch.Tensor):
    ndim = len(tensor.shape)
    if ndim > MAX_DIMS:
        raise ValueError(f"Only up to {MAX_DIMS} dimensions supported, got {ndim}")

    shape = list(tensor.shape) + [0] * (MAX_DIMS - ndim)
    return struct.pack(HEADER_FORMAT, ndim, *shape, tensor.element_size())

def load_header(header: bytes):
    ndim, d0, d1, d2, d3, dtype_code = struct.unpack('6i', header)
    assert 0 < ndim <= MAX_DIMS, f"ndim should be in (0, {MAX_DIMS}], got {ndim}"
    tensor_shape = (d0, d1, d2, d3)[:ndim]
    return tensor_shape, NP_DTYPE_MAP[dtype_code]

# todo: sending multiple tensors in one turn may lead to bottleneck in serializing tensors
# maybe adding one more function to pipeline gpu2cpu and transmission
def serialize_tensor(tensor: torch.Tensor):
    """
    prepare for transmission
    """
    tensor = tensor.contiguous()

    header = gen_header(tensor)

    if tensor.device.type == 'cuda':
        tensor_pinned = torch.empty_like(tensor, device='cpu', pin_memory=True)
        tensor_pinned.copy_(tensor, non_blocking=True)
        raw = tensor_pinned.numpy().tobytes()  # 隐式同步
        # raw = tensor.cpu().numpy().tobytes()
    else:
        raw = tensor.numpy().tobytes()
    
    return header, raw

def load_tensor(header: bytes, raw: bytes):
    """
    load from transmission
    """
    tensor_shape, dtype = load_header(header)

    # raw = socket.recv()
    arr = np.frombuffer(raw, dtype=dtype)
    if tensor_shape:
        arr = arr.reshape(tensor_shape)
    tensor = torch.from_numpy(arr)
    return tensor



class CommZMQ:
    def __init__(self, server_ip):
        self.context = zmq.Context()
        self.server_ip = server_ip
        self.serve_url = f'tcp://{server_ip}:{PORT}'
        self.is_server = self.server_ip in get_all_ips()
        # if self.is_server:
            # self.client_ID_trace = 0

    def send_tensor(self, socket: zmq.Socket, tensor: torch.Tensor, flags=0):
        tensor = tensor.contiguous()

        # if tensor.device.type == 'cuda':
        #     tensor_pinned = torch.empty_like(tensor, device='cpu', pin_memory=True)
        #     tensor_pinned.copy_(tensor, non_blocking=True)
        
        header = gen_header(tensor)

        # socket.send(header, flags | zmq.SNDMORE)

        if tensor.device.type == 'cuda':
            tensor_pinned = torch.empty_like(tensor, device='cpu', pin_memory=True)
            tensor_pinned.copy_(tensor, non_blocking=True)
            raw = tensor_pinned.numpy().tobytes()  # 隐式同步
        else:
            raw = tensor.numpy().tobytes()
        
        # socket.send_multipart(raw, flags, copy=False)
        socket.send_multipart([header, raw], copy=False)

    def recv_tensor(self, socket: zmq.Socket, device):
        # header = socket.recv()
        header, raw = socket.recv_multipart()
        tensor_shape, dtype = load_header(header)

        # raw = socket.recv()
        arr = np.frombuffer(raw, dtype=dtype)
        if tensor_shape:
            arr = arr.reshape(tensor_shape)
        tensor = torch.from_numpy(arr)
        return tensor.to(device)

    def close(self):
        if hasattr(self, "socket"):
            self.socket.close()
        self.context.term()


class CommPair(CommZMQ):
    def __init__(self, server_ip):
        super().__init__(server_ip)

        self.socket = self.context.socket(zmq.PAIR)
        # intialization
        if self.is_server:
            self.socket.bind(self.serve_url)
            msg = self.socket.recv()
            print(f'Recv from client: {msg}')
            self.socket.send_string(str(self.client_ID_trace))
        else:
            self.ip = get_ip_addr()
            self.socket.connect(self.serve_url)
            self.socket.send_string(self.ip)


class CommCS(CommZMQ):
    def __init__(self, server_ip, is_server=None):
        super().__init__(server_ip)

        self._running = True

        if is_server is not None:
            self.is_server = is_server

        self.send_queue = Queue()
        self.poller = zmq.Poller()

        if self.is_server:
            self.socket = self.context.socket(zmq.ROUTER)
            self.socket.bind(self.serve_url)
            self.poller.register(self.socket, zmq.POLLIN)
            # self.identifers = {}  # {identity: timestamp}
            self.recv_queues = {}  # {identity: Queue}
            self._start_threads()

        else:
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.connect(self.serve_url)
            self.poller.register(self.socket, zmq.POLLIN)
            # register after initialization
            self.identity = None
            self._register_client()
            self.recv_queue = Queue()
            self._start_threads()
        

    def _start_threads(self):
        if self.is_server:
            self.recv_thread = threading.Thread(target=self._serve_recv, daemon=True)
            self.send_thread = threading.Thread(target=self._serve_send, daemon=True)
        else:
            self.recv_thread = threading.Thread(target=self._keep_recv, daemon=True)
            self.send_thread = threading.Thread(target=self._keep_send, daemon=True)
        self.recv_thread.start()
        self.send_thread.start()

    def _register_client(self):
        try:
            self.socket.send_multipart([b'REG', b''])
            identity = self.socket.recv()
            self.identity = identity
            print(f"Client {identity} registered")
        except:
            print("Client registration failed")
            traceback.print_exc()
            sys.exit(1)

    def _handle_register(self, identity: bytes):
        self.recv_queues[identity] = Queue()
        self.socket.send_multipart([identity, identity])
        print(f"Client {identity} registered")


    def _handle_recv_tensor(self, identity, header, content):
        tensor = load_tensor(header, content)
        self.recv_queues[identity].put(tensor)

    def _handle(self, identity, header, content):
        timestamp = time.perf_counter()
        if header == b'REG':
            self._handle_register(identity)
        else:
            self._handle_recv_tensor(identity, header, content)

        # self.identifers[identity] = timestamp     
        
    def _serve_recv(self):
        """
        server: handling received messages
        """
        while self._running:
            try:
                events = dict(self.poller.poll(timeout=1000))
                if self.socket in events:
                    identity, header, content = self.socket.recv_multipart()
                    self._handle(identity, header, content)
            except:
                traceback.print_exc()
            # data = self.socket.recv_multipart()
            # identity, header, content = data
            # # timestamp = time.perf_counter()
            # self.handle(identity, header, content)

    def _keep_recv(self):
        """
        client: handling received tensor only
        """
        while self._running:
            try:
                events = dict(self.poller.poll(timeout=1000))
                if self.socket in events:
                    header, raw = self.socket.recv_multipart()
                    self.recv_queue.put(load_tensor(header, raw))
            except:
                traceback.print_exc()
            # header, raw = self.socket.recv_multipart()
            # self.recv_queue.put(load_tensor(header, raw))

    def _serve_send(self):
        """
        handling send messages
        """
        while self._running:
            try:
                identity, tensor = self.send_queue.get(timeout=1)
                header, raw = serialize_tensor(tensor)
                self.socket.send_multipart([identity, header, raw])
            except Empty:
                continue

    def _keep_send(self):
        while self._running:
            try:
                tensor = self.send_queue.get(timeout=1)
                header, raw = serialize_tensor(tensor)
                self.socket.send_multipart([header, raw])
            except Empty:
                continue

    def send_to(self, tensor, identity=None):
        """
        API for sending tensor: both client and server
        """
        self.send_queue.put(tensor if identity is None else (identity, tensor))

    def recv_from(self, identity=None, device=None):
        """
        API for receiving tensor: both client and server
        """
        if self.is_server:
            tensor = self.recv_queues[identity].get()
        else:
            tensor = self.recv_queue.get()
        
        return tensor if device is None else tensor.to(device)

    def stop(self):
        self._running = False
        self.send_thread.join()
        self.recv_thread.join()
        self.close()

