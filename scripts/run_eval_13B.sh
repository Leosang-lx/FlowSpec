#!/bin/bash

torchrun --nnodes=1 --master-port=12345 --nproc_per_node=5 run_pipe_eval.py \
--model_name llama2-13b \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-13b-chat-hf/new_stage_model_series_0+10+10+10+10_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-13B \
--extra_name llama2_13b_0521_seed &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nnodes=1 --master-port=12345 --nproc_per_node=5 run_pipe_eval.py \
--model_name vicuna-13b \
--base_model_dir /home/liux/big_file/pipeline_model/lmsys/vicuna-13b-v1.3/new_stage_model_series_0+10+10+10+10_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-Vicuna-13B-v1.3 \
--extra_name vicuna_13b_0521_seed &
SECOND_PID=$!

wait $SECOND_PID
kill -SIGINT "$SECOND_PID" 2>/dev/null || true
