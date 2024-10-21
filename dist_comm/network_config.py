import ipaddress

MAIN_WORKER_IP = '192.168.1.101'
# MAIN_WORKER_IP = '::1'
MASTER_IP = '192.168.1.150'
INTERFACE = 'eth0'
SUBNET = '192.168.1.1'
# server_ip = 'fe80::1d6b:9eb3:d29b:c7e0'
# server_ip = 'fe80::c597:1cbc:1f91:9817'
master_port = 9999
port_tcp = 8848
port_torch = 23456
SERVER_RANK = 0

DEFAULT_SIZE = 2


def ipv4_or_ipv6(ip):
    try:
        # 尝试将地址解析为 IPv4 地址
        ipaddress.IPv4Address(ip)
        return 4
    except ipaddress.AddressValueError:
        pass

    try:
        # 尝试将地址解析为 IPv6 地址
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
