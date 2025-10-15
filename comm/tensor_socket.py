import zmq
import torch
import numpy as np
import struct
import socket
from subprocess import getstatusoutput

DTYPE_MAP = {
    1: torch.int8,
    2: torch.float16,
    3: torch.int32,
    4: torch.int64,
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

class CommZMQ:
    def __init__(self, server_ip):
        self.context = zmq.socket()
        self.server_ip = server_ip
        self.serve_url = f'tcp://{server_ip}:{PORT}'
        self.is_server = self.server_ip in get_all_ips()
        if self.is_server:
            self.client_ID_trace = 0
    
    def bind(self):
        pass
    def connect(self):
        pass

    def send_tensor(self, tensor: torch.Tensor):
        pass

    def close(self):
        if hasattr(self, "socket"):
            self.socket.close()
        self.context.term()

class CommP2P(CommZMQ):
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


class CommClient:
    def __init__(self, server_ip):
        self.context = zmq.Context()
        self.serve_url = f'tcp://{server_ip}:{PORT}'
    def bind(self, server_ip, port):
        self.socket.bind(self.serve_url)
    
    def connect(self, client_ip, port):
        self.socket.connect(self.serve_url)
    
    def send_tensor():
        pass

