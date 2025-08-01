#!/bin/bash

CUDA_LAUNCH_BLOCKING=1

torchrun --nnodes=1 --master-port=22222 --nproc_per_node=5 tp/run_tp.py