from eagle.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
from transformers import AutoConfig
import time
from pipeline_utils import merge_two_tree, token_tree_partition
from profiler.profiler import prof

cache_dir = '/home/liux/LLM/models_hf/'

base_model_path = 'meta-llama/Llama-2-7b-chat-hf'
EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-7B'


base_model_path = cache_dir + base_model_path
EAGLE_model_path = cache_dir + EAGLE_model_path

config = AutoConfig.from_pretrained(base_model_path)

print(config)
# exit(0)

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

prompt = "Hello, how are you?"
input_ids = model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()

outputs, orig, hidden_states = model(input_ids, past_key_values=None, output_orig=True)
token = torch.argmax(orig[:, -1])[None, None]
input_ids_ea = torch.cat((input_ids, token), dim=1)

draft_tokens1, retrieve_indices1, tree_mask1, tree_position_ids1 = model.ea_layer.topK_genrate(
    hidden_states,
    input_ids_ea,
    model.base_model.lm_head,
    None,
    total_tokens=24,
    depth=1
)
model.ea_layer.reset_kv()
draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2 = model.ea_layer.topK_genrate(
    hidden_states,
    input_ids_ea,
    model.base_model.lm_head,
    None,
    total_tokens=64,
    depth=6
)

tokens_split, lens_split1, subseq_ri_cum_depths1 = token_tree_partition(
    draft_tokens1,
    retrieve_indices1,
    total_stage=3,
)
print(retrieve_indices1)
tree1 = (draft_tokens1.cpu(), retrieve_indices1.cpu(), tree_mask1.cpu(), tree_position_ids1.cpu())
tree2 = (draft_tokens2.cpu(), retrieve_indices2.cpu(), tree_mask2.cpu(), tree_position_ids2.cpu())
# warm up
for i in range(5):
    merge_two_tree(tree1, tree2, lens_split1, subseq_ri_cum_depths1)

for i in range(10):
    with prof.time_context(f"merge two trees", cpu=True):
        new_tree = merge_two_tree(tree1, tree2, lens_split1, subseq_ri_cum_depths1, prof=prof)

prof.print_all_events()

