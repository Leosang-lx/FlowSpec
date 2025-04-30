from eagle.ea_model import EaModel
from transformers import AutoConfig
import torch
import os
import json
from config.run_config import config as run_config

cache_dir = '/home/liux/LLM/models_hf/'
base_model_path = 'meta-llama/Llama-2-7b-chat-hf'
EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-7B'
base_model_path = cache_dir + base_model_path
EAGLE_model_path = cache_dir + EAGLE_model_path

config = AutoConfig.from_pretrained(base_model_path)

print(config)

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
    device_map="cuda:0",
    total_token=13,
    depth=3
)

print('EaMoel loaded')
model.eval()
tokenizer = model.tokenizer
prompt = "Hello, how are you?"
input_ids = tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()

outputs, orig, hidden_states = model(input_ids, past_key_values=None, output_orig=True)
token = torch.argmax(orig[:, -1])[None, None]
input_ids_ea = torch.cat((input_ids, token), dim=1)

draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_state = model.ea_layer.topK_genrate(
    hidden_states,
    input_ids_ea,
    model.base_model.lm_head,
    None,
    total_tokens=40,
    depth=5,
    log=True,
    return_last=True
)

last_tree = draft_tokens, retrieve_indices, tree_mask, tree_position_ids

draft_tokens, retrieve_indices, tree_mask, tree_position_ids, expand_size, last_state = model.ea_layer.expand_last(
    last_tree,
    last_state,
    model.base_model.lm_head,
    expand_depth=1,
    return_last=True
)





print(tokenizer.decode(draft_tokens[0]))
print()

