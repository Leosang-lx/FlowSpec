#!/usr/bin/env python3
import socket
import sys
import time

SERVER_PORT = 12345  # 必须与服务端一致
TIMEOUT = 5  # 超时时间（秒）

def test_tcp_connection(server_ip):
    print(f"[Client] 尝试连接 {server_ip}:{SERVER_PORT} ...")
    
    try:
        # 创建 socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        
        # 尝试连接
        start_time = time.time()
        sock.connect((server_ip, SERVER_PORT))
        connect_time = (time.time() - start_time) * 1000  # 毫秒
        
        # 接收服务端消息（可选）
        try:
            data = sock.recv(1024)
            if data:
                print(f"[Client] 收到服务端响应: {data.decode().strip()}")
        except:
            pass  # 无响应也正常
        
        sock.close()
        print(f"[Client] ✅ 连接成功！耗时 {connect_time:.2f} ms")
        print("success")
        return True

    except socket.timeout:
        print("[Client] ❌ 连接超时", file=sys.stderr)
        print("可能原因：", file=sys.stderr)
        print("  - 服务端未运行或未监听该端口", file=sys.stderr)
        print("  - 中间防火墙丢弃了连接请求（无响应）", file=sys.stderr)
        print("  - 网络路由不通", file=sys.stderr)

    except ConnectionRefusedError:
        print("[Client] ❌ 连接被拒绝 (Connection refused)", file=sys.stderr)
        print("可能原因：", file=sys.stderr)
        print("  - 服务端未在该端口监听", file=sys.stderr)
        print("  - 服务端防火墙明确拒绝了连接", file=sys.stderr)

    except OSError as e:
        if e.errno == 101 or "Network is unreachable" in str(e):
            print("[Client] ❌ 网络不可达 (Network unreachable)", file=sys.stderr)
            print("可能原因：", file=sys.stderr)
            print("  - 本地没有到目标IP的路由", file=sys.stderr)
            print("  - 目标IP地址不存在或不在同一网段且无网关", file=sys.stderr)
        elif e.errno == 113 or "No route to host" in str(e):
            print("[Client] ❌ 无路由到主机 (No route to host)", file=sys.stderr)
            print("可能原因：", file=sys.stderr)
            print("  - 目标主机不可达", file=sys.stderr)
            print("  - 中间路由器无法找到路径", file=sys.stderr)
        else:
            print(f"[Client] ❌ 网络错误: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[Client] ❌ 未知错误: {e}", file=sys.stderr)

    return False

def main():
    if len(sys.argv) != 2:
        print("用法: python3 tcp_client.py <server_ip>", file=sys.stderr)
        sys.exit(1)

    server_ip = sys.argv[1]
    
    # 简单校验 IP 格式（不严格）
    if server_ip.count('.') != 3:
        print("[Client] ❌ 无效的IP地址格式", file=sys.stderr)
        sys.exit(1)

    success = test_tcp_connection(server_ip)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()