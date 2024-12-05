import ipaddress
import datetime
import socket
import os

distributed = False

if distributed:
    MAIN_WORKER_IP = '192.168.1.104'
    ipvx = socket.AF_INET
    # Enable when using RaspberryPi
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    os.environ['GLOO_TIMEOUT'] = '20s'
else:
    MAIN_WORKER_IP = '::1'
    ipvx = socket.AF_INET6

MASTER_IP = '192.168.1.106'
INTERFACE = 'eth0'
SUBNET = '192.168.1'

master_port = 9999
port_tcp = 8848
port_torch = 23456
SERVER_RANK = 0
timeout_max = datetime.timedelta(seconds=10)

no_s = [104, 105, 126, 128]
ip_to_rank = {}
for rank, no in enumerate(no_s):
    ip_to_rank[f'{SUBNET}.{no}'] = rank

DEFAULT_SIZE = len(no_s) if distributed else 2


def ipv4_or_ipv6(ip):
    try:
        ipaddress.IPv4Address(ip)
        return 4
    except ipaddress.AddressValueError:
        pass

    try:
        ipaddress.IPv6Address(ip)
        return 6
    except ipaddress.AddressValueError:
        pass

    return 0


def gen_init_method(server_ip: str, port: int, protocol='tcp'):
    ipv = ipv4_or_ipv6(server_ip)
    if ipv == 4:
        init_method = f'{protocol}://{server_ip}:{port}'
    elif ipv == 6:
        init_method = f'{protocol}://[{server_ip}]:{port}'
    else:
        raise Exception('Invalid IP Address')

    return init_method
