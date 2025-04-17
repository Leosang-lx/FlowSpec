from dataclasses import dataclass

@dataclass
class Config:
    
    # model config
    base_model_dir: str = f"/home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/stage_model_series_6+9+9+8_half"
    EAGLE_model_path: str = "/home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    
    # run config
    your_message: str = "Hello"
    log: bool = True
    temperature: float = 0.5
    max_new_tokens: int = 512
    total_token: int = 64
    depth: int = 6
    
    # pipeline config
    pipeline_type: str = "naive"
    
    # device config
    device: str = "cuda"
    low_cpu_mem_usage: bool = True
    max_memory: str = "1GB"

config = Config()

