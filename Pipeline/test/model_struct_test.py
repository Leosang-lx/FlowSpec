print(f"__package__: {__package__}")

from eagle.ea_model import EaModel

from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from fastchat.model import get_conversation_template
from datetime import timedelta
import torch
import torch.distributed as dist

from safetensors.torch import load_file

def model_struct_test():
    cache_dir = '/home/liux/LLM/models_hf/'

    base_model_path = 'meta-llama/Llama-2-7b-chat-hf'
    EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-7B'

    base_model_path = cache_dir + base_model_path
    EAGLE_model_path = cache_dir + EAGLE_model_path

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        device_map="cuda:0",
        total_token=16,
        depth=2
    )

    state_dict = load_file(base_model_path + '/model-00001-of-00002.safetensors', device=f"cuda:0")
    print("--------------------------------\n state_dict\n--------------------------------")
    print(state_dict.keys())
    print(state_dict['model.embed_tokens.weight'])

    print("--------------------------------\n model\n--------------------------------")
    print(model.state_dict().keys())
    print("--------------------------------\n base_model\n--------------------------------")
    print(model.base_model.state_dict().keys())
    print(model.base_model.state_dict()['model.embed_tokens.weight'])
    print("--------------------------------\n ea_layer\n--------------------------------")
    print(model.ea_layer.state_dict().keys())

    base_model_path = f"/home/liux/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/stage_model_series_8+8+8+8/stage_model_0"
    EAGLE_model_path = "/home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"

    stage_model = StageEaModel.from_pretrained(
                stage_base_model_path=base_model_path,
                ea_model_path=EAGLE_model_path,
                torch_dtype=torch.float16,
                # low_cpu_mem_usage=True,
                device_map=f"cuda:0",
                total_token=16,
                depth=2,
            )
    # state_dict = load_file(base_model_path + '/model.safetensors', device=f"cuda:0")
    state_dict = torch.load(base_model_path + '/pytorch_model.bin')
    print("--------------------------------\n stage_state_dict\n--------------------------------")
    print(state_dict.keys())
    print(state_dict['model.embed_tokens.weight'])
    print("--------------------------------\n stage_model\n--------------------------------")
    print(stage_model.state_dict().keys())
    print("--------------------------------\n stage_model.base_model\n--------------------------------")
    print(stage_model.stage_base_model.state_dict().keys())
    print(stage_model.stage_base_model.state_dict()['model.embed_tokens.weight'])
    print("--------------------------------\n stage_model.ea_layer\n--------------------------------")
    print(stage_model.ea_layer.state_dict().keys())

