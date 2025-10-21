from comm.tensor_socket import CommCS
import time
import torch
from datetime import datetime
import argparse
import multiprocessing as mp


num_rounds = 5
tensor_size = (100, 4096)
device = torch.device('cuda:0')


def server_process():
    test_tensor = torch.randn(tensor_size, dtype=torch.float16).to(device)

    """服务器进程"""
    print(f"[SERVER] Starting server...")
    
    # 服务器IP设置为本地
    server = CommCS("127.0.0.1", is_server=True)
    
    # 等待客户端连接
    time.sleep(1)
    
    print(f"[SERVER] Server started and waiting for client")
    
    # 存储接收到的张量和时间戳
    received_tensors = []
    receive_times = []
    sleep_intervals = []
    
    # 获取客户端身份
    client_identity = None
    for identity in server.recv_queues.keys():
        client_identity = identity
        break
    
    # # 创建测试张量
    # tensor_size = (100, 100)
    # test_tensor = torch.randn(tensor_size)
    
    for round_idx in range(num_rounds):

        print(f"\n[SERVER] Round {round_idx + 1}/{num_rounds}")

        # 发送张量到客户端
        send_time = time.time()
        print(f"[SERVER] Sending tensor at: {datetime.fromtimestamp(send_time)} ({send_time:.3f})")
        server.send_to(test_tensor, client_identity)

        
        # 记录sleep开始时间
        sleep_start = time.time()
        print(f"[SERVER] Sleep start: {datetime.fromtimestamp(sleep_start)} ({sleep_start:.3f})")
        
        # Sleep 0.5秒
        time.sleep(0.5)
        
        # 记录sleep结束时间
        sleep_end = time.time()
        print(f"[SERVER] Sleep end: {datetime.fromtimestamp(sleep_end)} ({sleep_end:.3f}) (duration: {sleep_end - sleep_start:.3f}s)")
        sleep_intervals.append((sleep_start, sleep_end))
        
        # 尝试接收客户端的张量
        try:
            received_tensor = server.recv_from(client_identity, device)
            if received_tensor is not None:
                receive_time = time.time()
                print(f"[SERVER] Received tensor at: {datetime.fromtimestamp(receive_time)} ({receive_time:.3f}), shape: {received_tensor.shape}")
                received_tensors.append(received_tensor)
                receive_times.append(receive_time)
            else:
                print("[SERVER] Timeout waiting for tensor from client")
        except Exception as e:
            print(f"[SERVER] Error receiving tensor: {e}")
    
    print(f"\n[SERVER] Finished. Received {len(received_tensors)} tensors")
    
    # 打印详细的时间信息
    print("\n[SERVER] Time Analysis:")
    for i, (tensor, recv_time) in enumerate(zip(received_tensors, receive_times)):
        sleep_start, sleep_end = sleep_intervals[i] if i < len(sleep_intervals) else (0, 0)
        time_after_sleep = recv_time - sleep_end if sleep_end > 0 else 0
        print(f"[SERVER] Round {i+1}: Received at {datetime.fromtimestamp(recv_time)} (after sleep: {time_after_sleep:.6f}s)")
    
    server.stop()
    print("[SERVER] Server closed")

def client_process():
    test_tensor = torch.randn(tensor_size, dtype=torch.float16).to(device)

    """客户端进程"""
    print(f"[CLIENT] Starting client, connecting to server...")
    
    # 客户端连接到本地服务器
    client = CommCS("127.0.0.1", is_server=False)
    
    # 存储接收到的张量和时间戳
    received_tensors = []
    receive_times = []
    sleep_intervals = []
    
    # # 创建测试张量
    # tensor_size = (100, 100)
    # test_tensor = torch.randn(tensor_size)
    # num_rounds = 5
    
    for round_idx in range(num_rounds):
        print(f"\n[CLIENT] Round {round_idx + 1}/{num_rounds}")
        
        # 发送张量到服务器
        send_time = time.time()
        print(f"[CLIENT] Sending tensor to server at: {datetime.fromtimestamp(send_time)} ({send_time:.3f})")
        client.send_to(test_tensor)
        
        # 记录sleep开始时间
        sleep_start = time.time()
        print(f"[CLIENT] Sleep start: {datetime.fromtimestamp(sleep_start)} ({sleep_start:.3f})")
        
        # Sleep 0.5秒
        time.sleep(0.5)
        
        # 记录sleep结束时间
        sleep_end = time.time()
        print(f"[CLIENT] Sleep end: {datetime.fromtimestamp(sleep_end)} ({sleep_end:.3f})")
        sleep_intervals.append((sleep_start, sleep_end))
        

        # 首先尝试接收服务器发送的张量
        try:
            received_tensor = client.recv_from(device=device)
            if received_tensor is not None:
                receive_time = time.time()
                print(f"[CLIENT] Received tensor from server at: {datetime.fromtimestamp(receive_time)} ({receive_time:.3f}), shape: {received_tensor.shape}")
                
                received_tensors.append(received_tensor)
                receive_times.append(receive_time)
            else:
                print("[CLIENT] No tensor received from server in this round")
        except Exception as e:
            print(f"[CLIENT] Error receiving tensor: {e}")
    
    print(f"\n[CLIENT] Finished. Sent {num_rounds} tensors, received {len(received_tensors)} tensors")
    
    # 打印详细的时间信息
    print("\n[CLIENT] Time Analysis:")
    for i, (tensor, recv_time) in enumerate(zip(received_tensors, receive_times)):
        sleep_start, sleep_end = sleep_intervals[i] if i < len(sleep_intervals) else (0, 0)
        time_after_sleep = recv_time - sleep_end if sleep_end > 0 else 0
        print(f"[CLIENT] Round {i+1}: Received at {datetime.fromtimestamp(recv_time)} (after sleep: {time_after_sleep:.6f}s)")
    
    client.stop()
    print("[CLIENT] Client closed")


if __name__ == "__main__":
    # 创建进程
    server_process = mp.Process(target=server_process)
    client_process = mp.Process(target=client_process)
    
    # 启动进程
    server_process.start()
    client_process.start()
    
    # 等待进程结束
    server_process.join()
    client_process.join()