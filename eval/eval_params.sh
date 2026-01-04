#!/bin/bash

torchrun --nnodes=1 --master-port=12345 --nproc_per_node=5 eval_params.py \
--extra_name turns_1201
SECOND_PID=$!

wait $SECOND_PID
kill -SIGINT "$SECOND_PID" 2>/dev/null || true