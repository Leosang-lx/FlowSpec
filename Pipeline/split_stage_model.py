from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from stage_modeling_llama import StageLlamaModel, StageLlamaModelForCausalLM
from eagle.ea_model import EaModel
from eagle.cnets import Model
from transformers import AutoConfig
from pipeline_utils import split_close_equal
from typing import Tuple
import torch
import os

cache_dir = '/home/liux/LLM/models_hf/'

base_model_path = 'meta-llama/Llama-2-7b-chat-hf'
EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-7B'

base_model_path = cache_dir + base_model_path
EAGLE_model_path = cache_dir + EAGLE_model_path


def gen_stage_model_config_series(total_stage: int, base_ea_config) -> StageEaConfig:
    assert isinstance(total_stage, int) and total_stage > 0
    total_hidden_layers = base_ea_config.num_hidden_layers
    hidden_layers_split = split_close_equal(total_hidden_layers, total_stage)
    print(f'total_hidden_layers={total_hidden_layers}, total_stage={total_stage}, hidden_layers_split={hidden_layers_split}')
    stage_model_config_series = []
    for stage, hidden_layer_num in enumerate(hidden_layers_split):
        has_embedding = True if stage == 0 or stage == total_stage - 1 else False
        has_draft_model = True if stage == 0 else False
        has_lm_head = True if stage == 0 or stage == total_stage - 1 else False  

        stage_model_config = StageEaConfig(
            ea_config=base_ea_config,
            stage=stage,
            stage_num_hidden_layers_list=hidden_layers_split,
            base_model_name_or_path=base_model_path,
            has_embedding=has_embedding,
            has_draft_model=has_draft_model,
            has_lm_head=has_lm_head,
        )
        stage_model_config_series.append(stage_model_config)

    return stage_model_config_series


def gen_stage_model(base_ea_model: EaModel, stage_model_config: StageEaConfig, save_dir=None):
    """
    save_dir: the directory of according to the base model type on the server/master
    e.g., a valid example: save_dir = '~/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf'
    """
    base_model = base_ea_model.base_model
    embedding_layer = base_model.model.embed_tokens
    hidden_layers = base_model.model.layers

    start_hidden_layers, end_hidden_layers = stage_model_config.layer_range
    print(f'start_hidden_layers={start_hidden_layers}, end_hidden_layers={end_hidden_layers}')
    partial_hidden_layers = hidden_layers[start_hidden_layers: end_hidden_layers]

    embedding = embedding_layer if stage_model_config.has_embedding else None
    print(f'stage_model_config.has_embedding={stage_model_config.has_embedding}')
    stage_model = StageLlamaModel(
        stage_model_config,
        embed_tokens=embedding,
        hidden_layers=partial_hidden_layers,
        post_init=False # why?
    )
    stage_model.eval()
    print(f"stage_model.config.is_last_stage={stage_model.config.is_last_stage}, stage_model_config.has_lm_head={stage_model_config.has_lm_head}")
    stage_model_with_LMHead = StageLlamaModelForCausalLM(
        stage_model_config,
        stage_model=stage_model,
    )

    ea_layer = base_ea_model.ea_layer if stage_model_config.has_draft_model else None

    if save_dir is None:
        stage_ea_model = StageEaModel(stage_model_with_LMHead, base_model_path, stage_model_config, ea_layer)
        return stage_ea_model

    else:
        stage = stage_model_config.stage
        joined_list = '+'.join(map(str, stage_model_config.stage_num_hidden_layers_list))
        stage_model_dir = os.path.join(save_dir, f'stage_model_series_{joined_list}/stage_model_{stage}')
        print(f'Save stage_model_{stage} to:' + stage_model_dir)
        print('--Saving stage_model_config...')
        stage_model_config.save_pretrained(stage_model_dir)
        print('--Saving stage_model...')
        stage_model_with_LMHead.save_pretrained(stage_model_dir)
        print('--Done!')


def analy_state_dict(state_dict: dict):
    mem_usage = 0
    for key, param_tensor in state_dict.items():
        num_elements = param_tensor.numel()
        bytes_per_element = param_tensor.element_size()
        mem_usage += num_elements * bytes_per_element
        print(f'{key}: {param_tensor.shape}, {bytes_per_element}B each')

    print(f'{mem_usage / (1 << 30):.2f} GB')


def gen_stage_model_series(base_ea_model: EaModel, stage_model_config_series, save_dir=None):
    """
    Save all necessary files for a pipelined stage model series **about the base model**
    - tokenizer file
    - for each stage model
      - config.json
      - pytorch_model.bin
    the draft model should be loaded additionally
    An example for the directory structure is shown below:
    ~/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf -> save_dir
    |--tokenizer_file
    |--draft_model_dir
    |--stage_model_series_8+8+8+8  # config.stage_num_hidden_layers_list
       |--stage_model_1
          |--config.json
          |--pytorch_model.bin
       |--stage_model_2
          |--config.json
          |--pytorch_model.bin
       |--...
    |--stage_base_model_series2
    |--...
    """
    stage_model_list = []
    for stage_model_config in stage_model_config_series:
        stage_model = gen_stage_model(base_ea_model, stage_model_config, save_dir)
        if save_dir is None:
            stage_model_list.append(stage_model)
    if save_dir is None:
        return stage_model_list


def load_stage_model(
        cache_dir='/home/liux/LLM',
        base_model_tag='meta-llama/Llama-2-7b-chat-hf',
        draft_model_tag='yuhuili/EAGLE-llama2-chat-7B',
        stage_model_series='8+8+8+8',
        stage=-1,
        torch_dtype=torch.float16
):
    assert isinstance(stage, int) and stage > -1
    stage_model_cache_dir = os.path.join(
        cache_dir,
        'pipeline_model',
        base_model_tag,
        f'stage_model_series_{stage_model_series}',
        f'stage_model_{stage}'
    )

    if not os.path.exists(stage_model_cache_dir):
        raise FileNotFoundError('Cache directory for stage model not found!')
    stage_model_config = StageEaConfig.from_pretrained(stage_model_cache_dir)
    ea_model_path = os.path.join(
        cache_dir,
        'models_hf',
        draft_model_tag
    ) if stage_model_config.has_draft_model else None
    stage_model = StageEaModel.from_pretrained(
        stage_base_model_path=stage_model_cache_dir,
        ea_model_path=ea_model_path,
        torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        # device_map="cuda:0",
        total_token=16,
        depth=2
    )
    return stage_model

def compare_embeddings(base_model_embeddings, stage_model_embeddings):
    """
    Compare embeddings between base model and stage model to ensure they match.
    
    Returns:
        bool: True if embeddings match, False otherwise
    """
    if base_model_embeddings is None or stage_model_embeddings is None:
        print("One of the embedding layers is None, cannot compare")
        return False
    
    # Check if shapes match
    if base_model_embeddings.weight.shape != stage_model_embeddings.weight.shape:
        print(f"Embedding shapes don't match: {base_model_embeddings.weight.shape} vs {stage_model_embeddings.weight.shape}")
        return False
    
    # Check if values are identical
    is_equal = torch.allclose(base_model_embeddings.weight, stage_model_embeddings.weight, rtol=1e-5, atol=1e-8)
    if not is_equal:
        print("Embedding values don't match!")
        # Calculate differences
        diff = (base_model_embeddings.weight - stage_model_embeddings.weight).abs()
        print(f"Max difference: {diff.max().item()}")
        print(f"Mean difference: {diff.mean().item()}")
    else:
        print("Embeddings match perfectly!")
    
    return is_equal

if __name__ == '__main__':

    # [start] test generate stage model
    base_ea_config = AutoConfig.from_pretrained(base_model_path)
    
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        # device_map="auto",
        # total_token=-1,
        total_token=16,
        depth=2
    )
    stage_model = load_stage_model(stage=0)
    base_model_embeddings = model.base_model.model.embed_tokens.to('cuda:0')
    stage_model_embeddings = stage_model.stage_base_model.model.embed_tokens.to('cuda:0')
    compare_embeddings(base_model_embeddings, stage_model_embeddings)
    
    # print('Generating stage model config series...')
    # stage_model_config_series = gen_stage_model_config_series(4, base_ea_config)
    
    # print('============bsae_model===========')
    # analy_state_dict(model.base_model.state_dict())
    
    # print('============stage_model===========')
    # print(f'stage_model_config_series[0]={stage_model_config_series[0]}')
    # print(f'stage_model_config_series[0].has_embedding={stage_model_config_series[0].has_embedding}')
    # stage_model = gen_stage_model(model, stage_model_config_series[0], None)
    # analy_state_dict(stage_model.stage_base_model.state_dict())
    # # torch.save(stage_model.state_dict(), 'stage_model_test.pth')
    # exit(0)
    
    # stage1_model_config = stage_model_configs[0]
    # stage1_model = gen_stage_model(model, stage1_model_config)
    # stage_model_save_dir = '/home/liux/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf'
    # print(f'save_dir={stage_model_save_dir}')
    
    # gen_stage_model_series(model, stage_model_config_series, stage_model_save_dir)
    # exit(0)

    # [start] test load stage model
    # stage_model = load_stage_model(stage=3)
    # analy_state_dict(stage_model.state_dict())










