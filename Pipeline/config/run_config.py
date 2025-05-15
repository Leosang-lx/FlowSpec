from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    
    model_name: str = "vicuna"
    
    # network config
    hardware: str = "server" # "jetson" or "server"
    if hardware == "jetson":
        set_network: bool = False
        password: str = "nvidia"
        interface: str = "eth0"
        rate_mbps: float = 150
        delay_ms: float = 0.0

    # run config
    mode = "demo" # "eval" or "demo"
    pipeline_type: str = "pruned"
    
    if mode == "eval":  
        warmup = True
        warmup_repeat = 5
        test_repeat = 1 # this refer to num of choices in the eval set
    else:
        warmup = True
        warmup_repeat = 10
        test_repeat = 10
    
    eval_record: bool = True
    log: bool = False
    prof: bool = False
    save_timestamps: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 512
    
    timeout: int = 15
    
    # model config
    if model_name == "llama2":
        if hardware == "server":
            base_model_dir: str = f'/home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16'
            EAGLE_model_path: str = "/home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
        else:
            base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16"
            EAGLE_model_path: str = f"/home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    elif model_name == "vicuna":
        if hardware == "server":
            base_model_dir: str = f'/home/liux/big_file/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16'
            EAGLE_model_path: str = "/home/liux/big_file/vicuna/EAGLE-Vicuna-7B-v1.3"
        else:
            base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16"
            EAGLE_model_path: str = f"/home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3"
    
    # eval config
    if mode == "eval":
        dataset_name: str = "mt_bench"
        question_path: str = "./data/" + dataset_name + "/question.jsonl"
        question_begin: int = 30
        question_end: int = 50
        
    # demo config
    if mode == "demo":
        your_message: str = "Hello"
    # pipeline config
    if pipeline_type == "naive":
        draft_gen_sort_score: bool = False
        num_stage: int = 5 # 5 or 4
        init_total_token: int = 80
        init_topk: int = 10
        init_depth: int = 6
    
    if pipeline_type == "pruned":
        draft_gen_sort_score: bool = True
        num_stage: int = 5 # 5 or 4
        init_total_token: int = 80
        init_topk: int = 10
        init_depth: int = 6
        init_subseq_token: int = 16
        
    if pipeline_type == "continuous":
        draft_gen_sort_score: bool = True
        num_stage: int = 5 # only support 5 stage now
        
        # draft config
        init_total_token: int = 80
        init_topk: int = 10
        init_depth: int = 6
        init_subseq_token: int = 16
        
        # expand draft config
        expand_total_token: int =64
        expand_topk: int =10 # now must be the same as init_topk
        expand_depth: int =6
        expand_subseq_token: int = -1
        
        none_expand: bool = True
        if none_expand:
            none_expand_size: int = 48
            none_expand_depth: int = 1

    if pipeline_type == "pipedec":
        init_total_token: int = 64 # this parameter is meaningless in pipedec
        init_topk: int = 16
        init_depth: int = 1 # meaningless in pipedec as well
        
    # device config
    device: str = "cuda"
    low_cpu_mem_usage: bool = True
    max_memory: str = "1GB"

config = Config()

