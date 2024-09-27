import ipaddress

server_ip = '192.168.1.101'
# server_ip = 'fe80::1d6b:9eb3:d29b:c7e0'
# server_ip = 'fe80::c597:1cbc:1f91:9817'
port = 8848
server_rank = 0


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


init_method = None
ipv = ipv4_or_ipv6(server_ip)
if ipv == 4:
    init_method = f'tcp://{server_ip}:{port}'
elif ipv == 6:
    init_method = f'tcp://[{server_ip}]:{port}'
else:
    raise Exception('Invalid IP Address')
