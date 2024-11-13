import torch
import torch.nn.functional as F
from time import perf_counter_ns
import numpy as np

b = 1
d_model = 4096
intermediate_dimension = 11008

x = torch.randn(b, d_model)  # , dtype=torch.float16)
print('dtype of x:', x.dtype)
W = torch.randn(d_model, intermediate_dimension)  # , dtype=torch.float16)
print('dtype of W:', W.dtype)
bias = torch.zeros(intermediate_dimension)

types = ['@', 'matmul', 'addmm', 'nn']


def computation_task(type: str):  # only the computation process without other initialization
    if type == '@':
        y = x @ W
    elif type == 'matmul':
        y = torch.matmul(x, W)
    elif type == 'addmm':
        y = torch.addmm(bias, x, W)
    elif type == 'nn':
        y = F.linear(x, W.T)
    else:
        raise Exception('Unknown matmul type!')


def test_latency(function, args):
    start = perf_counter_ns()

    y = function(args)

    end = perf_counter_ns()
    return end - start


def latency_test():
    for mm_type in types:
        latencies = []
        for i in range(10):
            latencies.append(test_latency(computation_task, mm_type)/1e6)
        print(f'type:{mm_type}\n- ls:{latencies}\n- avg:{np.mean(latencies)}ms')


if __name__ == '__main__':
    latency_test()
