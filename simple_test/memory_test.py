import gc
import os
import psutil
import torch
from transformers.activations import NewGELUActivation
from memory_profiler import profile
import objgraph

torch.no_grad()

@profile
def main():
    x = torch.randn(1, 101, 384, dtype=torch.float32)


    act = NewGELUActivation()
    objgraph.show_most_common_types()


    x = act(x)
    gc.collect()
    a = 1
    objgraph.show_most_common_types()

main()


