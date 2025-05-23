#!/bin/bash

torchrun --nnodes=1 --master-port=12345 --nproc_per_node=5 run_pipe_eval.py \
--model_name llama2 \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--extra_name llama2_0520_seed &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nnodes=1 --master-port=12345 --nproc_per_node=5 run_pipe_eval.py \
--model_name vicuna \
--base_model_dir /home/liux/big_file/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/big_file/vicuna/EAGLE-Vicuna-7B-v1.3 \
--extra_name vicuna_0520 &
SECOND_PID=$!

wait $SECOND_PID
kill -SIGINT "$SECOND_PID" 2>/dev/null || true
