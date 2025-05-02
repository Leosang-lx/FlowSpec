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

# print(config)

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
    total_tokens=20,
    depth=3,
    log=True,
    return_last=True
)
print(f'========Draft tokens========')
print(f'draft_tokens: {draft_tokens}')
print(f'retrieve_indices: {retrieve_indices}')
# print(f'tree_mask: {tree_mask}')
print(f'tree_position_ids: {tree_position_ids}')
print(f'retrieve_position_ids: {tree_position_ids[retrieve_indices]}')

last_tree = draft_tokens, retrieve_indices, tree_mask, tree_position_ids

draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_state = model.ea_layer.expand_last(
    last_tree,
    last_state,
    model.base_model.lm_head,
    None,
    hidden_states.device,
    expand_depth=1,
    expand_size=10,
    return_last=True
)

print(f'========Expanded tokens========')
print(f'draft_tokens: {draft_tokens}')
print(f'retrieve_indices: {retrieve_indices}')
# print(f'tree_mask: {tree_mask}')
print(f'tree_position_ids: {tree_position_ids}')
print(f'expand_size: {20}')
# print(f'last_state: {last_state}')
print(f'retrieve_position_ids: {tree_position_ids[retrieve_indices]}')

assert draft_tokens.size(-1) == retrieve_indices.max()+1 == tree_mask.size(-1) == tree_mask.size(-2) == tree_position_ids.size(-1)


print(tokenizer.decode(draft_tokens[0]))
print()

