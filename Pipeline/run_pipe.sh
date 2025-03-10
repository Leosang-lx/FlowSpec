#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=4 run_pipe.py