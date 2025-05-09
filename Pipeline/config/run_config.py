from dataclasses import dataclass

@dataclass
class Config:
    
    # run config
    mode = "demo" # "eval" or "demo"
    pipeline_type: str = "continuous"
    warmup = True
    warmup_repeat = 10
    test_repeat = 10 # this refer to num of choices in the eval set
    
    log: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 512
    
    # model config
    base_model_dir: str = f'/home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16'
    # base_model_dir: str = f'/home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+10+11+11_fp16'
    EAGLE_model_path: str = "/home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    
    # base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16"
    # EAGLE_model_path: str = f"/home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    
    # eval config
    if mode == "eval":
        dataset_name: str = "mt_bench"
        question_path: str = "./data/" + dataset_name + "/question.jsonl"
        question_begin: int = 10
        question_end: int = 20
        
    # demo config
    if mode == "demo":
        your_message: str = "Hello"

    # pipeline config
    if pipeline_type == "naive":
        init_total_token: int = 64
        init_topk: int = 10
        init_depth: int = 6
    
    if pipeline_type == "pruned":
        init_total_token: int = 64
        init_topk: int = 10
        init_depth: int = 6
    
    if pipeline_type == "continuous":
        # draft config
        init_total_token: int = 160
        init_topk: int = 10
        init_depth: int = 8
        init_subseq_token: int = 32
        
        # expand draft config
        expand_total_token: int = 64
        expand_topk: int =10 # now must be the same as init_topk
        expand_depth: int =6
        expand_subseq_token: int = -1
        
        none_expand: bool = False
        if none_expand:
            none_expand_size: int = 48
            none_expand_depth: int = 1
        
    # device config
    device: str = "cuda"
    low_cpu_mem_usage: bool = True
    max_memory: str = "1GB"

config = Config()

