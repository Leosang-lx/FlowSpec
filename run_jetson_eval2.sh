#! /bin/bash

NODE_RANK=0
WORLD_SIZE=5
NPROC_PER_NODE=1
MASTER_ADDR="192.168.1.161"
MASTER_PORT="12345"


torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name llama2 \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--temperature 0.0 \
--pipeline_type "naive" \
--extra_name navie_0_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name llama2 \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--temperature 0.0 \
--pipeline_type "pipedec" \
--extra_name pipedec_0_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name llama2 \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--temperature 0.0 \
--pipeline_type "continuous" \
--extra_name continuous_0_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name llama2 \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--temperature 1.0 \
--pipeline_type "naive" \
--extra_name navie_1_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name llama2 \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--temperature 1.0 \
--pipeline_type "pipedec" \
--extra_name pipedec_1_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name llama2 \
--base_model_dir /home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B \
--temperature 1.0 \
--pipeline_type "continuous" \
--extra_name continuous_1_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name vicuna \
--base_model_dir /home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3 \
--temperature 0.0 \
--pipeline_type "naive" \
--extra_name navie_0_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name vicuna \
--base_model_dir /home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3 \
--temperature 0.0 \
--pipeline_type "pipedec" \
--extra_name pipedec_0_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name vicuna \
--base_model_dir /home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3 \
--temperature 0.0 \
--pipeline_type "continuous" \
--extra_name continuous_0_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name vicuna \
--base_model_dir /home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3 \
--temperature 1.0 \
--pipeline_type "naive" \
--extra_name navie_1_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name vicuna \
--base_model_dir /home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3 \
--temperature 1.0 \
--pipeline_type "pipedec" \
--extra_name pipedec_1_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_pipe_eval3.py \
--model_name vicuna \
--base_model_dir /home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16 \
--EAGLE_model_path /home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3 \
--temperature 1.0 \
--pipeline_type "continuous" \
--extra_name continuous_1_0519 &
FIRST_PID=$!

wait $FIRST_PID
kill -SIGINT "$FIRST_PID" 2>/dev/null || true
