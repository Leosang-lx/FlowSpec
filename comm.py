import asyncio
import pickle
import time
from queue import SimpleQueue
import socket
import struct
from torch import Tensor
from subprocess import getstatusoutput
from sys import getsizeof
import psutil

__buffer__ = 65535
__timeout__ = 10


def list_network_interfaces():
    # 获取所有网络接口的信息
    addrs = psutil.net_if_addrs()

    # 遍历所有网络接口
    for interface, addresses in addrs.items():
        print(f"Interface: {interface}")
        for addr in addresses:
            print(f"  - Address: {addr.address}")
            print(f"  - Netmask: {addr.netmask}")
            print(f"  - Broadcast: {addr.broadcast}")


# list_network_interfaces()


def ping(dest: str, ping_cnt: int, timeout: int):  # '1' means connected, else 0
    assert 0 < ping_cnt < 10
    status, output = getstatusoutput(
        f"ping {dest} -c {ping_cnt} -w {timeout} | tail -2 | head -1 | awk {{'print $4'}}")  # send one packet and check the recv packet
    print(f'ping result {output}')
    if status == 0 and int(output) > 0:
        return True  # ping得通
    else:
        return False


data_header_format = 'I'
data_header_size = struct.calcsize(data_header_format)


def gen_bytes(obj):
    data = pickle.dumps(obj)
    data_size = len(data)
    header = struct.pack(data_header_format, data_size)
    return header + data


def accept_connection(server_socket: socket, recv_list: list, stop, timeout=None):
    while True:
        if stop():
            break
        try:
            conn, addr = server_socket.accept()
            conn.settimeout(timeout)
            recv_list.append((conn, addr[0]))  # only ip
            print(f'Recv connection from {addr}')
        except socket.timeout:
            continue
        except Exception as e:
            print(e)


def accept_n_connections(server_socket: socket, n: int, timeout=None):
    recv_conns = []
    for _ in range(n):
        try:
            conn, addr = server_socket.accept()
            conn.settimeout(timeout)
            recv_conns.append((conn, addr[0]))  # only ip
            print(f'Recv connection from {addr}')
        except socket.timeout:
            raise Exception(f'Fail to recv {n} connections')
    assert len(recv_conns) == n
    return recv_conns


def put_recv_data(conn: socket, recv_queue: SimpleQueue):
    try:
        data = recv_data(conn)
        recv_queue.put(data)
    except socket.timeout:
        print('Recv data time out')
    except Exception as e:
        print(e)


def connect_to_other(ip_list: list, port: int, socket_list: list, self_ip):
    try:
        for i, worker_ip in enumerate(ip_list):
            if worker_ip != self_ip:
                addr = worker_ip, port
                conn = socket.create_connection(addr, timeout=5)
                print(f'Connected to {worker_ip}')
                socket_list.append((conn, addr[0]))
    except socket.timeout:
        print('Create connections timeout')
    except Exception as e:
        print(e)


# def connect_to_other_local(port_list: list, socket_list: list, self_ip):
#     try:
#         for i, port in enumerate(port_list):
#             if worker_ip != self_ip:
#                 addr = worker_ip, port
#                 conn = create_connection(addr, timeout=5)
#                 print(f'Connected to {worker_ip}')
#                 socket_list.append((conn, addr[0]))
#     except timeout:
#         print('Create connections timeout')
#     except Exception as e:
#         print(e)


def send_data(send_socket: socket, data, waiting=0):
    if not isinstance(data, (bytes, bytearray)):
        data = pickle.dumps(data)
    data_size = len(data)
    header = struct.pack(data_header_format, data_size)
    if waiting:
        time.sleep(waiting)
    res = send_socket.sendall(header + data)
    return res


async def async_send_data(send_socket: socket, data, waiting=0):
    if not isinstance(data, (bytes, bytearray)):
        data = pickle.dumps(data)
    data_size = len(data)
    header = struct.pack(data_header_format, data_size)
    if waiting:
        await asyncio.sleep(waiting)  # 1.sleep before sending
    res = send_socket.sendall(header + data)
    return res


def recv_data(recv_socket: socket):
    msg = recv_socket.recv(data_header_size, socket.MSG_WAITALL)
    header = struct.unpack(data_header_format, msg)
    data_size = header[0]
    data = bytearray(data_size)
    ptr = memoryview(data)
    while data_size:
        nrecv = recv_socket.recv_into(buffer=ptr, nbytes=min(4096, data_size))
        ptr = ptr[nrecv:]
        data_size -= nrecv
        # recv = recv_socket.recv(min(4096, data_size), MSG_WAITALL)  # deprecated
        # data += recv
        # data_size -= len(recv)
    data = pickle.loads(data)
    return data


async def async_recv_data(recv_socket: socket):
    msg = recv_socket.recv(data_header_size)
    header = struct.unpack(data_header_format, msg)
    data_size = header[0]
    data = bytearray(data_size)
    ptr = memoryview(data)
    # data = b''
    while data_size:
        nrecv = recv_socket.recv_into(buffer=ptr, nbytes=min(4096, data_size), flags=socket.MSG_WAITALL)
        ptr = ptr[nrecv:]
        data_size -= nrecv
        # recv = recv_socket.recv(min(4096, data_size))
        # data += recv
        # data_size -= len(recv)
    return pickle.loads(data)


__format__ = 'IHHH'  # header: (data size, layer num, left range, right range]
__size__ = struct.calcsize(__format__)


def send_tensor(send_socket: socket, data: Tensor, layer_num: int, data_range: tuple[int, int]):
    """
    send the padding data of intermediate result of certain layer to other device
    :param send_socket: the socket of UDP protocol to send data
    :param data: the (Tensor type) data to send
    :param layer_num: the intermediate result of which layer
    :param data_range: the corresponding range of data to the layer output
    :return: the sending status
    """
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    header = struct.pack(__format__, data_size, layer_num, *data_range)
    res = send_socket.sendall(header + serialized)
    return res


async def async_send_tensor(send_socket: socket, data: Tensor, layer_num: int, data_range: tuple[int, int]):
    serialized = pickle.dumps(data)
    data_size = len(serialized)
    print(f'send output from layer {layer_num} of size {data_size / 1024}KB')
    header = struct.pack(__format__, data_size, layer_num, *data_range)
    res = send_socket.sendall(header + serialized)
    return res


def recv_tensor(recv_socket: socket):
    data = b''
    header_size = __size__
    while header_size:
        recv = recv_socket.recv(header_size)
        data += recv
        header_size -= len(recv)
    header = struct.unpack(__format__, data)
    data_size = header[0]
    data = b''
    while data_size:
        recv = recv_socket.recv(min(4096, data_size))
        data += recv
        data_size -= len(recv)
    return (header[1], (header[2], header[3])), pickle.loads(data)  # (layer_num, range) data


async def async_recv_tensor(recv_socket: socket):
    data = b''
    header_size = __size__
    while header_size:
        recv = recv_socket.recv(header_size)
        data += recv
        header_size -= len(recv)
    header = struct.unpack(__format__, data)
    data_size = header[0]
    data = b''
    while data_size:
        recv = recv_socket.recv(min(4096, data_size))
        data += recv
        data_size -= len(recv)
    return (header[1], (header[2], header[3])), pickle.loads(data)  # (layer_num, range) data
