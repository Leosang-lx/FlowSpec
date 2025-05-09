#!/bin/bash

torchrun --nnodes=1 --master-port=12345 --nproc_per_node=5 run_pipe.py