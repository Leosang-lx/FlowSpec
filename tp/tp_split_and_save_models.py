# run with PYTHONPATH=. python tp/tp_split_and_save_models.py

from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from model.stage_modeling_llama import StageLlamaModel, StageLlamaModelForCausalLM
from eagle.ea_model import EaModel
from eagle.cnets import Model
from transformers import AutoConfig
from pipeline_utils import split_close_equal
from typing import Tuple
import torch
import os
import copy
from safetensors.torch import save_file
from test.model_struct_test import model_struct_test
import hashlib
import torch.nn as nn

cache_dir = '/home/liux/big_file/'

# base_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# EAGLE_model_path = 'yuhuili/EAGLE-LLaMA3-Instruct-8B'
# base_model_path = 'lmsys/vicuna-13b-v1.3'
# EAGLE_model_path = 'yuhuili/EAGLE-Vicuna-13B-v1.3'
# base_model_path = 'meta-llama/Llama-2-13b-chat-hf'
# EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-13B'
# base_model_name = 'meta-llama/Llama-2-7b-chat-hf'
# EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-7B'
base_model_name = 'lmsys/vicuna-7b-v1.3'
EAGLE_model_path = 'yuhuili/EAGLE-Vicuna-7B-v1.3'

base_model_path = cache_dir + base_model_name
EAGLE_model_path = cache_dir + EAGLE_model_path

def gen_stage_model_config_series(split_cnt: int, base_ea_config) -> StageEaConfig:
    assert isinstance(split_cnt, int) and split_cnt > 0
    stage_model_config_series = []
    for stage in range(split_cnt):
        # [update] only draft stage has draft model
        has_draft_model = stage == 0
        # [update] only the first stage has embedding and lm_head
        has_embedding = True
        has_lm_head = True

        stage_model_config = StageEaConfig(
            ea_config=base_ea_config,
            stage=stage,  # [udpate] starts from 0, but rank starts from 1
            stage_num_hidden_layers_list=[0,32, 32, 32, 32],
            base_model_name_or_path=base_model_path,
            has_embedding=has_embedding,
            has_draft_model=has_draft_model,
            has_lm_head=has_lm_head,
        )
        if stage != 0:
            stage_model_config.num_attention_heads = stage_model_config.num_attention_heads//(split_cnt-1)
            stage_model_config.num_key_value_heads = stage_model_config.num_key_value_heads//(split_cnt-1)
        stage_model_config_series.append(stage_model_config)

    return stage_model_config_series

def slice_weight_for_tp(value: torch.Tensor, key: str, tp_rank: int, tp_size: int):
    if value.ndim < 2:  # norm, rotary 等不切
        return value
    if any(k in key for k in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'embed_tokens', 'lm_head']):
        print(f"[{key}] original={value.shape} -> sliced={value.chunk(tp_size, dim=0)[tp_rank-1].shape}")
        return value.chunk(tp_size, dim=0)[tp_rank-1].contiguous()  # output dim 切分
    elif any(k in key for k in ['o_proj', 'down_proj']):
        print(f"[{key}] original={value.shape} -> sliced={value.chunk(tp_size, dim=1)[tp_rank-1].shape}")
        return value.chunk(tp_size, dim=1)[tp_rank-1].contiguous()  # input dim 切分
    else:
        return value  # 默认不切
    
def save_stage_dict(base_ea_model: EaModel, config: StageEaConfig, save_dir: str, draft_stage: bool = True):
    """
    save_dir: the directory of according to the base model type on the server/master
    e.g., a valid example: save_dir = '~/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf'
    """
    state_dict = base_ea_model.base_model.state_dict()
    stage_state_dict = {}

    if config.has_embedding:
        for key, value in state_dict.items():
            if key.startswith('model.embed_tokens'):
                if config.is_draft_stage:
                    stage_state_dict[key] = value
                else:
                    stage_state_dict[key] = slice_weight_for_tp(value, key, config.stage, config.total_stage-1)
    
    if not config.is_draft_stage:
        for key, value in state_dict.items():
            if key.startswith('model.layers.'):
                stage_state_dict[key] = slice_weight_for_tp(value, key, config.stage, config.total_stage-1)
            
    if config.has_lm_head:
        for key, value in state_dict.items():
            if key.startswith('lm_head'):
                if config.is_draft_stage:
                    stage_state_dict[key] = value
                else:
                    stage_state_dict[key] = slice_weight_for_tp(value, key, config.stage, config.total_stage-1)
    
    if not config.is_draft_stage:
        for key, value in state_dict.items():
            if key.startswith('model.norm'):
                stage_state_dict[key] = slice_weight_for_tp(value, key, config.stage, config.total_stage-1)
        
    # print(f'stage_state_dict={stage_state_dict}')
    stage = config.stage
    stage_file_name = ('new_' if draft_stage else '') + f'stage_model_series_tp' + ('_fp16' if base_ea_model.base_model.dtype == torch.float16 else '') + f'/stage_model_{stage}'
    stage_file_dir = os.path.join(save_dir, stage_file_name)
    print(f'Save stage_model_{stage} to:' + stage_file_dir)
    print('--Saving stage_model_config...')
    config.save_pretrained(stage_file_dir)
    print('--Saving stage_model...')
    # torch.save(stage_state_dict, stage_file_dir + '/pytorch_model.bin')
    save_file(stage_state_dict, stage_file_dir + '/model.safetensors', metadata={'format': 'pt'})
    print('--Done!')
    
if __name__ == '__main__':
    base_ea_config = AutoConfig.from_pretrained(base_model_path)

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16
    )
    # exit(0)
    
    stage_model_config_series = gen_stage_model_config_series(5, base_ea_config)
    
    stage_model_save_dir = f'/home/liux/big_file/tp_model/{base_model_name}'
    for config in stage_model_config_series:
        save_stage_dict(model, config, stage_model_save_dir, draft_stage=True)