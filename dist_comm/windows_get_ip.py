import socket


def get_ip_address():
    # 获取主机名
    hostname = socket.gethostname()

    # 获取与主机名关联的所有 IP 地址
    ip_addresses = socket.gethostbyname_ex(hostname)[2]

    # 通常会返回多个 IP 地址（例如，本地回环地址 127.0.0.1 和实际的网络接口地址）
    # 我们可以过滤掉本地回环地址，只保留实际的网络接口地址
    ip_addresses = [ip for ip in ip_addresses if not ip.startswith("127.") and not ip.startswith("::1")]

    if ip_addresses:
        return ip_addresses  # 返回第一个非本地回环地址
    else:
        return None


if __name__ == "__main__":
    ip_address = get_ip_address()
    if ip_address:
        print(f"Your IP address is: {ip_address}")
    else:
        print("Could not determine the IP address.")