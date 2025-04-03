import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from typing import Dict, Tuple, Optional
import time

class CommHandler:
    def __init__(self, max_workers: int = 4):
        """
        初始化通信处理器
        :param max_workers: 线程池最大线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 发送队列 (data, dst_rank)
        self.send_queue = queue.Queue()
        self.broadcast_send_queue = queue.Queue()
        
        # 接收队列字典 {src_rank: queue}
        self.recv_queues: Dict[int, queue.Queue] = {}
        self.recv_queues_lock = threading.Lock()
        
        # 线程控制标志
        self._running = False
        self._threads = []
        
    def start(self):
        """启动所有通信线程"""
        if self._running:
            return
            
        self._running = True
        
        # 启动发送线程
        self._threads.append(self.executor.submit(self._send_worker))
        self._threads.append(self.executor.submit(self._broadcast_send_worker))
        
        # 启动接收线程
        with self.recv_queues_lock:
            for src_rank in self.recv_queues.keys():
                self._threads.append(
                    self.executor.submit(self._recv_worker, src_rank)
                )
    
    def stop(self):
        """停止所有通信线程"""
        self._running = False
        for future in self._threads:
            future.cancel()
        self.executor.shutdown(wait=True)
        
    def send(self, tensor: torch.Tensor, dst_rank: int):
        """
        发送张量到指定rank
        :param tensor: 要发送的张量
        :param dst_rank: 目标rank
        """
        self.send_queue.put((tensor, dst_rank))
        
    def broadcast_send(self, tensor: torch.Tensor):
        """
        广播发送张量
        :param tensor: 要广播的张量
        """
        self.broadcast_send_queue.put(tensor)
        
    def register_recv_queue(self, src_rank: int):
        """
        注册接收队列
        :param src_rank: 来源rank
        """
        with self.recv_queues_lock:
            if src_rank not in self.recv_queues:
                self.recv_queues[src_rank] = queue.Queue()
                if self._running:
                    # 如果已经运行，则启动新的接收线程
                    self._threads.append(
                        self.executor.submit(self._recv_worker, src_rank)
                    )
                    
    def recv(self, src_rank: int, timeout: Optional[float] = None) -> torch.Tensor:
        """
        从指定rank接收张量
        :param src_rank: 来源rank
        :param timeout: 超时时间(秒)
        :return: 接收到的张量
        """
        try:
            with self.recv_queues_lock:
                recv_queue = self.recv_queues[src_rank]
            return recv_queue.get(timeout=timeout)
        except KeyError:
            raise ValueError(f"No receive queue registered for src_rank {src_rank}")
            
    def _send_worker(self):
        """发送线程工作函数"""
        while self._running:
            try:
                tensor, dst_rank = self.send_queue.get(timeout=0.1)
                
                # 发送头信息 (shape, dtype)
                header = (tensor.shape, str(tensor.dtype))
                dist.send(torch.tensor(0), dst=dst_rank)  # 发送信号
                dist.send(header, dst=dst_rank)
                
                # 发送张量数据
                dist.send(tensor, dst=dst_rank)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Send worker error: {e}")
                
    def _broadcast_send_worker(self):
        """广播发送线程工作函数"""
        while self._running:
            try:
                tensor = self.broadcast_send_queue.get(timeout=0.1)
                
                # 广播头信息
                header = (tensor.shape, str(tensor.dtype))
                dist.broadcast(torch.tensor(0), src=dist.get_rank())  # 发送信号
                dist.broadcast(header, src=dist.get_rank())
                
                # 广播张量数据
                dist.broadcast(tensor, src=dist.get_rank())
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Broadcast send worker error: {e}")
                
    def _recv_worker(self, src_rank: int):
        """接收线程工作函数"""
        while self._running:
            try:
                # 接收信号
                signal = torch.zeros(1)
                dist.recv(signal, src=src_rank)
                
                # 接收头信息
                header = None
                dist.recv(header, src=src_rank)
                shape, dtype_str = header
                
                # 创建空张量接收数据
                tensor = torch.zeros(shape, dtype=getattr(torch, dtype_str))
                dist.recv(tensor, src=src_rank)
                
                # 放入接收队列
                with self.recv_queues_lock:
                    self.recv_queues[src_rank].put(tensor)
                    
            except Exception as e:
                if self._running:  # 只打印非关闭导致的错误
                    print(f"Receive worker for src_rank {src_rank} error: {e}")
                time.sleep(0.1)  # 避免CPU占用过高