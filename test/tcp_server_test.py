#!/usr/bin/env python3
import socket
import sys

SERVER_PORT = 12345  # 可自定义端口

def start_server():
    try:
        # 创建 TCP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定到所有接口的指定端口
        sock.bind(('0.0.0.0', SERVER_PORT))
        sock.listen(1)
        
        local_ip = socket.gethostbyname(socket.gethostname())
        print(f"[Server] 正在监听端口 {SERVER_PORT}（本机IP可能为 {local_ip}，请确保客户端使用正确的IP）")
        print(f"[Server] 等待客户端连接...")

        conn, addr = sock.accept()
        print(f"[Server] 收到来自 {addr[0]}:{addr[1]} 的连接")

        # 发送一个简单确认消息
        conn.send(b"TCP connection OK\n")
        conn.close()
        sock.close()
        print("[Server] 连接已处理并关闭")

    except Exception as e:
        print(f"[Server] 启动失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    start_server()