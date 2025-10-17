import zmq
import torch
import numpy as np
import struct
import socket
from subprocess import getstatusoutput
from queue import Queue, Empty

DTYPE_MAP = {
    1: torch.int8,
    2: torch.float16,
    3: torch.int32,
    4: torch.int64,
}

NP_DTYPE_MAP = {

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
    return tensor_shape, DTYPE_MAP[dtype_code]


class CommZMQ:
    def __init__(self, server_ip):
        self.context = zmq.socket()
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
    def __init__(self, server_ip):
        super().__init__(server_ip)

        if self.is_server:
            self.socket = self.context.socket(zmq.ROUTER)
            self.socket.bind(self.serve_url)
            self.identifers = {}  # {identity: timestamp}
            self.send_queues = Queue()
        else:
            self.socket = self.context(zmq.DEALER)
            self.socket.connect(self.serve_url)
        
    def serve(self):
        while True:
            identity,  = self.socket.recv_multi_part()

    def send_from_server(self, identity, tensor):
        self.socket.send_multi_part([identity, ten], copy=False)

    def register_client(self):
        pass
    def register_server(self):
        pass



