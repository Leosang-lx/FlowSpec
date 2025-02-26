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

base_ea_config = AutoConfig.from_pretrained(base_model_path)
# print(base_ea_config.rope_theta)
# exit(0)

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    # total_token=-1,
    total_token=64,
    depth=6
)


def gen_stage_model_config(total_stage: int, hidden_layers_split) -> StageEaConfig:
    assert isinstance(total_stage, int) and total_stage > 0
    total_hidden_layers = base_ea_config.num_hidden_layers
    hidden_layers_split = split_close_equal(total_hidden_layers, total_stage)

    stage_model_config_series = []
    for stage, hidden_layer_num in enumerate(hidden_layers_split):
        has_embedding = True if stage == 0 else False
        has_draft_model = True if stage == 0 else False
        # has_lm_head = True if stage == 0 else False  

        stage_model_config = StageEaConfig(
            ea_config=base_ea_config,
            stage=stage,
            stage_num_hidden_layers_list=hidden_layers_split,
            base_model_name_or_path=base_model_path,
            has_embedding=has_embedding,
            has_draft_model=has_draft_model,
            has_lm_head=has_embedding,
        )
        stage_model_config_series.append(stage_model_config)
    
    return stage_model_config_series

print('Generating stage model config series...')
stage_model_config_series = gen_stage_model_config(4, base_ea_config)
# stage_model_config_series[0].save_pretrained('./')
# exit(0)

def gen_stage_model(base_ea_model: EaModel, stage_model_config, save_dir=None):
    """
    save_dir: the directory of according to the base model type on the server/master
    e.g., a valid example: save_dir = '~/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf'
    """
    base_model = base_ea_model.base_model
    embedding_layer = base_model.model.embed_tokens
    hidden_layers = base_model.model.layers

    start_hidden_layers, end_hidden_layers = stage_model_config.layer_range
    print(stage_model_config.layer_range)
    partial_hidden_layers = hidden_layers[start_hidden_layers: end_hidden_layers]
    
    embedding = embedding_layer if stage_model_config.has_embedding else None
    stage_model = StageLlamaModel(
        stage_model_config,
        embed_tokens=embedding,
        hidden_layers=partial_hidden_layers,
        post_init=False
    )
    stage_model.eval()

    stage_model_with_LMHead = StageLlamaModelForCausalLM(
        stage_model_config,
        stage_model=stage_model,
    )

    ea_layer = base_ea_model.ea_layer

    if save_dir is None:
        stage_ea_model = StageEaModel(stage_model_with_LMHead, base_model_path, stage_model_config, ea_layer)
        return stage_ea_model
    
    else:
        joined_list = '+'.join(map(str, stage_model_config.stage_num_hidden_layers_list))
        stage_model_dir = os.path.join(save_dir, f'stage_model_series_{joined_list}/stage_model_{stage_model_config.stage}')
        print('save to:' + stage_model_dir)
        stage_model_config.save_pretrained(stage_model_dir)
        stage_model_with_LMHead.save_pretrained(stage_model_dir)

stage_model = gen_stage_model(model, stage_model_config_series[0], None)
print(stage_model.state_dict())
exit(0)
        

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
    |--stage_model_series_[8,8,8,8]  # config.stage_num_hidden_layers_list
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
        

# stage1_model_config = stage_model_configs[0]
# stage1_model = gen_stage_model(model, stage1_model_config)
stage_model_save_dir = '/home/liux/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf'
print(f'save_dir={stage_model_save_dir}')


gen_stage_model_series(model, stage_model_config_series, save_dir=None)




        

    




