from dataclasses import dataclass

@dataclass
class Config:
    
    #mode
    mode = "eval" # "eval" or "demo"
    warmup = True
    warmup_repeat = 5
    test_repeat = 10
    
    # model config
    base_model_dir: str = f'/home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16'
    EAGLE_model_path: str = "/home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    
    # base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/stage_model_series_6+9+9+8_half"
    # EAGLE_model_path: str = "/home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    
    # run config
    your_message: str = "Hello"
    log: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 512

    # pipeline config
    pipeline_type: str = "continuous"
    
    if pipeline_type == "naive":
        init_total_token: int = 64
        init_topk: int = 10
        init_depth: int = 6
    
    if pipeline_type == "pruned":
        init_total_token: int = 160
        init_topk: int = 10
        init_depth: int = 8
    
    if pipeline_type == "continuous":
        # draft config
        init_total_token: int = 160
        init_topk: int = 10
        init_depth: int = 8
        init_subseq_token: int = 32
        
        # expand draft config
        expand_total_token: int = 64
        expand_topk: int = 10 # now must be the same as init_topk
        expand_depth: int = 6
        expand_subseq_token: int = 64
        
    # device config
    device: str = "cuda"
    low_cpu_mem_usage: bool = True
    max_memory: str = "1GB"

config = Config()

