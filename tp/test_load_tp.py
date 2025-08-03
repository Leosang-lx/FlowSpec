from tp_ea_model import TPEaModel
from tp_modeling_llama import TPLlamaForCausalLM
from stage_ea_config import StageEaConfig
from accelerate import init_empty_weights
import torch
rank = 4
tp_base_model_path= f"/home/liux/big_file/tp_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_tp_fp16/stage_model_{rank}"

device = 0
with init_empty_weights():
    tp_base_model = TPLlamaForCausalLM.from_pretrained(
            tp_base_model_path, tp_rank=4, tp_size=4
        )
    # tp_model = TPEaModel.from_pretrained(
    #         tp_base_model_path=f"/home/nvidia/LLM/tp_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_tp_fp16/stage_model_{rank}",
    #         ea_model_path=None,
    #         torch_dtype=torch.float16,
    #         low_cpu_mem_usage=True,
    #         # max_memory={"cpu": "1GB"},
    #         use_safetensors=True,
    #         # quantization_config=run_config.quant_config,
    #         # device_map=f"cuda:{device}",
    #         # total_token=run_config.init_total_token,
    #         # depth=run_config.init_depth,
    #         # top_k=run_config.init_topk if run_config.pipeline_type != "pipedec" else run_config.init_topk_pipedec,
    #     )
    # print()
    # st = tp_base_model.state_dict()
    # for key, value in st.items():
    #     print(key, value.shape)



