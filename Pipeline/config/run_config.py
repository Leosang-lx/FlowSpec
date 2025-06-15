from dataclasses import dataclass, field
from typing import List
import time
from transformers import BitsAndBytesConfig
import torch

@dataclass
class Config:
    
    model_name: str = "llama2"
    
    # network config
    hardware: str = "server" # "jetson" or "server"
    if hardware == "jetson":
        set_network: bool = False
        password: str = "nvidia"
        interface: str = "eth0"
        rate_mbps: float = 150
        delay_ms: float = 0.0

    # run config
    mode = "eval" # "eval" or "demo"
    
    if mode == "eval":  
        pipeline_types: List[str] = field(default_factory=lambda: ["naive", "continuous", "pipedec"])
        
        warmup = True
        warmup_repeat = 5
        test_repeat = 1 # this refer to num of choices in the eval set
        error_repeat = 3
        change_seed = False
        
        # dataset_names: List[str] = field(default_factory=lambda: ["mt_bench", "humaneval", "gsm8k", "alpaca", "sum", "qa"])
        dataset_names: List[str] = field(default_factory=lambda: ["mt_bench"])
        question_paths: List[str] = field(init=False)
        question_begin: int = 30
        question_end: int = 31
        
        eval_record: bool = True
        
        temperatures: List[float] = field(default_factory=lambda: [0.0, 1.0])
    else:
        pipeline_type: str = "continuous"
        
        warmup = False
        warmup_repeat = 10
        test_repeat = 1
        
        your_message: str = "who are you?"
        
        temperature: float = 0.0
    
    
    log: bool = False
    prof: bool = False
    save_timestamps: bool = False
    max_new_tokens: int = 128
    
    timeout: int = 15
    
    # model config
    if model_name == "llama2":
        if hardware == "server":
            base_model_dir: str = f'/home/liux/big_file/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16'
            EAGLE_model_path: str = "/home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
        else:
            base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_0+8+8+8+8_fp16"
            EAGLE_model_path: str = f"/home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    elif model_name == "llama2-13b":
        if hardware == "server":
            base_model_dir: str = '/home/liux/big_file/pipeline_model/meta-llama/Llama-2-13b-chat-hf/new_stage_model_series_0+10+10+10+10_fp16'
            EAGLE_model_path: str = "/home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-13B"
        else:
            base_model_dir: str = f'/home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-13b-chat-hf/new_stage_model_series_0+10+10+10+10_fp16'
            EAGLE_model_path: str = "/home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-13B"
    elif model_name == "vicuna":
        if hardware == "server":
            base_model_dir: str = f'/home/liux/big_file/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16'
            EAGLE_model_path: str = "/home/liux/big_file/vicuna/EAGLE-Vicuna-7B-v1.3"
        else:
            base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/vicuna/Vicuna-7B-v1.3/new_stage_model_series_0+8+8+8+8_fp16"
            EAGLE_model_path: str = f"/home/nvidia/LLM/vicuna/EAGLE-Vicuna-7B-v1.3"
    elif model_name == "vicuna-13b":
        if hardware == "server":
            base_model_dir: str = f'/home/liux/big_file/pipeline_model/lmsys/vicuna-13b-v1.3/new_stage_model_series_0+10+10+10+10_fp16'
            EAGLE_model_path: str = "/home/liux/LLM/models_hf/yuhuili/EAGLE-Vicuna-13B-v1.3"
        else:
            base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/vicuna/Vicuna-13B-v1.3/new_stage_model_series_0+10+10+10+10_fp16"
            EAGLE_model_path: str = "/home/nvidia/LLM/models_hf/yuhuili/EAGLE-Vicuna-13B-v1.3"

    quant = False
    quant_config = BitsAndBytesConfig(  # Fastest parameters with 4-bit quantization
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4"
        ) if quant else None
    
    # pipeline config
    if mode == "eval":
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
        
        none_expand: bool = False
        if none_expand:
            none_expand_size: int = 48
            none_expand_depth: int = 1
        
        init_topk_pipedec: int = 16
    else:
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
            
            none_expand: bool = False
            if none_expand:
                none_expand_size: int = 48
                none_expand_depth: int = 1

        if pipeline_type == "pipedec":
            init_total_token: int = 64 # this parameter is meaningless in pipedec
            init_topk_pipedec: int = 16
            init_depth: int = 1 # meaningless in pipedec as well
        
    # device config
    device: str = "cuda"
    low_cpu_mem_usage: bool = True
    max_memory: str = "1GB"
    
    def __post_init__(self):
        if self.mode == "eval":
            self.question_paths = ["./data/" + dataset_name + "/question.jsonl" for dataset_name in self.dataset_names]

config = Config()

