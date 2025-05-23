#! /bin/bash

NODE_RANK=0
WORLD_SIZE=5
NPROC_PER_NODE=1
MASTER_ADDR="192.168.1.161"
MASTER_PORT="12345"


torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval.py \
--model_name llama2 \
--base_model_dir /home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--extra_name llama2_0520 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT $FIRST_PID 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval.py \
--model_name vicuna \
--base_model_dir /home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3 \
--extra_name vicuna_0520 &
SECOND_PID=$!

wait $SECOND_PID
kill -SIGINT $SECOND_PID 2>/dev/null || true

