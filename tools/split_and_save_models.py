# run with PYTHONPATH=. python tools/split_and_save_models.py

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

base_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
EAGLE_model_path = 'yuhuili/EAGLE-LLaMA3-Instruct-8B'
# base_model_path = 'lmsys/vicuna-13b-v1.3'
# EAGLE_model_path = 'yuhuili/EAGLE-Vicuna-13B-v1.3'
# base_model_path = 'meta-llama/Llama-2-13b-chat-hf'
# EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-13B'

base_model_path = cache_dir + base_model_name
EAGLE_model_path = cache_dir + EAGLE_model_path

def gen_stage_model_config_series(split_cnt: int, base_ea_config) -> StageEaConfig:
    assert isinstance(split_cnt, int) and split_cnt > 0
    total_hidden_layers = base_ea_config.num_hidden_layers
    hidden_layers_split = [0] + split_close_equal(total_hidden_layers, split_cnt)
    print(f'total_hidden_layers={total_hidden_layers}, hidden_layers_split={hidden_layers_split}')
    stage_model_config_series = []
    for stage, hidden_layer_num in enumerate(hidden_layers_split):
        # [update] only draft stage has draft model
        has_draft_model = stage == 0
        # [update] only the first stage has embedding and lm_head
        has_embedding = stage == 1
        has_lm_head = stage == 0

        stage_model_config = StageEaConfig(
            ea_config=base_ea_config,
            stage=stage,  # [udpate] starts from 0, but rank starts from 1
            stage_num_hidden_layers_list=hidden_layers_split,
            base_model_name_or_path=base_model_path,
            has_embedding=has_embedding,
            has_draft_model=has_draft_model,
            has_lm_head=has_lm_head,
        )
        stage_model_config_series.append(stage_model_config)

    return stage_model_config_series

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
                stage_state_dict[key] = value
    
    for i in range(*config.layer_range):
        # stage_key = f'model.layers.{i % config.stage_num_hidden_layers_list[config.stage]}.'
        stage_key = f'model.layers.{i - config.layer_range[0]}.'
        for key, value in state_dict.items():
            if key.startswith(f'model.layers.{i}.input_layernorm'):
                stage_state_dict[stage_key + 'input_layernorm.weight'] = value
            elif key.startswith(f'model.layers.{i}.mlp.down_proj'):
                stage_state_dict[stage_key + 'mlp.down_proj.weight'] = value
            elif key.startswith(f'model.layers.{i}.mlp.gate_proj'):
                stage_state_dict[stage_key + 'mlp.gate_proj.weight'] = value
            elif key.startswith(f'model.layers.{i}.mlp.up_proj'):
                stage_state_dict[stage_key + 'mlp.up_proj.weight'] = value
            elif key.startswith(f'model.layers.{i}.post_attention_layernorm'):
                stage_state_dict[stage_key + 'post_attention_layernorm.weight'] = value
            elif key.startswith(f'model.layers.{i}.self_attn.k_proj'):
                stage_state_dict[stage_key + 'self_attn.k_proj.weight'] = value
            elif key.startswith(f'model.layers.{i}.self_attn.o_proj'):
                stage_state_dict[stage_key + 'self_attn.o_proj.weight'] = value
            elif key.startswith(f'model.layers.{i}.self_attn.q_proj'):
                stage_state_dict[stage_key + 'self_attn.q_proj.weight'] = value
            elif key.startswith(f'model.layers.{i}.self_attn.rotary_emb'):
                stage_state_dict[stage_key + 'self_attn.rotary_emb.inv_freq'] = value
            elif key.startswith(f'model.layers.{i}.self_attn.v_proj'):
                stage_state_dict[stage_key + 'self_attn.v_proj.weight'] = value
            
    if config.has_lm_head:
        for key, value in state_dict.items():
            if key.startswith('lm_head'):
                stage_state_dict[key] = value
    
    if config.is_last_stage:
        for key, value in state_dict.items():
            if key.startswith('model.norm'):
                stage_state_dict[key] = value
    
    # print(f'stage_state_dict={stage_state_dict}')
    stage = config.stage
    joined_list = '+'.join(map(str, config.stage_num_hidden_layers_list))
    stage_file_name = ('new_' if draft_stage else '') + f'stage_model_series_{joined_list}' + ('_fp16' if base_ea_model.base_model.dtype == torch.float16 else '') + f'/stage_model_{stage}'
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
    
    stage_model_config_series = gen_stage_model_config_series(4, base_ea_config)
    
    stage_model_save_dir = f'/home/liux/big_file/pipeline_model/{base_model_name}'
    for config in stage_model_config_series:
        save_stage_dict(model, config, stage_model_save_dir, draft_stage=True)