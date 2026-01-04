#! /bin/bash

NODE_RANK=0
WORLD_SIZE=5
NPROC_PER_NODE=1
MASTER_ADDR="192.168.1.161"
MASTER_PORT="12345"


torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval.py \
--extra_name test_depths
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT $FIRST_PID 2>/dev/null || true