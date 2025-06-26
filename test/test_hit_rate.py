from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
from transformers import AutoConfig
import time
from eagle.model.utils import calculate_model_size_with_buffers, prepare_logits_processor
from eagle.model.kv_cache import *

cache_dir = '/home/liux/LLM/models_hf/'

base_model_path = 'meta-llama/Llama-2-7b-chat-hf'
EAGLE_model_path = 'yuhuili/EAGLE-llama2-chat-7B'
   
base_model_path = cache_dir + base_model_path
EAGLE_model_path = cache_dir + EAGLE_model_path

torch.cuda.set_device(1) 
config = AutoConfig.from_pretrained(base_model_path)

print(config)
# exit(0)

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
    device_map="cuda:1",
    total_token=64,
    depth=6,
)

model_size = calculate_model_size_with_buffers(model)
print(f"Model size: {model_size:.2f} MB")

model.eval()
your_message="Hello"
# conv = get_conversation_template("vicuna")

conv = get_conversation_template("llama-2-chat")
sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
conv.system_message = sys_p

conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt() + " "
print('\n=========PROMPT=========')
print(prompt)

input_ids=model.tokenizer([prompt]).input_ids
print('Input lenght:', len(input_ids[0]))
input_ids = torch.as_tensor(input_ids).cuda()

# base_model = model.base_model
draft_model = model.ea_layer
input_len = input_ids.size(-1)

def compare_base_draft(input_ids, temperature=0.0, top_k=0.0, top_p=0.0, topk_draft=10):
    # initialization
    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        logits_processor = None
    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data
    # prefill of base model
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )
    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        # print(f'logits: {logits.shape}, {logits.dtype}')
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        # print(f"probabilities: {probabilities}")
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    # prefill of draft model
    input_ids = input_ids.to(hidden_states.device)
    input_ids = input_ids[:, 1:]
    input_ids = input_ids.to(hidden_states.device)
    draft_model.reset()
    # topk_draft = 10
    input_len = input_ids.shape[1]
    
    test_cnt = 0
    from typing import Iterable
    if isinstance(topk_draft, Iterable):
        many_topk = True
        topk_size = sum(1 for _ in topk_draft)
        hit_cnt = [0] * topk_size
    else:
        many_topk = False
        hit_cnt = 0

    while True:
        # new token from base model
        last_hidden_states = hidden_states
        if current_length_data[0].item() == 0:
            outputs, orig, hidden_states = model(
                input_ids, past_key_values=past_key_values, output_orig=True
            )
        else:
            output, orig, hidden_states = model(
                token, past_key_values=past_key_values, output_orig=True
            )
        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(None, logits)
            # print(f'logits: {logits.shape}, {logits.dtype}')
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            # print(f"probabilities: {probabilities}")
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(orig[:, -1])
            token = token[None, None]
        
        # next draft from draft model
        if hasattr(draft_model, "stable_kv") and draft_model.stable_kv is not None:
            kv_len = draft_model.stable_kv[0][0].shape[2]
            # print(f'kv_len={kv_len}, input_ids:{input_ids.size(-1)}, hidden_states:{hidden_states.size(-2)}')
            out_hidden, draft_key_values = draft_model(last_hidden_states, input_ids=input_ids[:, kv_len:],
                                                past_key_values=draft_model.stable_kv, use_cache=True)
        else:
            out_hidden, draft_key_values = draft_model(last_hidden_states, input_ids=input_ids, use_cache=True)
        draft_model.stable_kv = draft_key_values
        last_hidden = out_hidden[:, -1]
        last_headout = model.base_model.lm_head(last_hidden)
        last_p = draft_model.logsoftmax(last_headout)

        # test
        test_cnt += 1
        if many_topk:
            for i, topk in enumerate(topk_draft):
                top = torch.topk(last_p, topk, dim=-1)
                topk_index, topk_p = top.indices, top.values
                if torch.isin(token, topk_index).item():
                    hit_cnt[i] += 1
        else:
            top = torch.topk(last_p, topk_draft, dim=-1)
            topk_index, topk_p = top.indices, top.values
            if torch.isin(token, topk_index).item():
                hit_cnt += 1
        
        # update input_ids
        input_ids = torch.cat((input_ids, token), dim=-1)
        # judge exit
        # if is_llama3:
        #     if stop_token_id in input_ids[0, input_len:].tolist():
        #         break

        if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if test_cnt > 512:
            break
        if input_ids.shape[1] > 2048:
            break
    # print(f'Hit rate: {hit_cnt/test_cnt*100:.2f}%')
    return input_ids, test_cnt, hit_cnt

topk_draft = [i for i in range(1, 11)]
output_ids, new_token, hit_cnt = compare_base_draft(input_ids, temperature=0.5, topk_draft=topk_draft)
print(f'test_cnt: {new_token}')
print(f'topk_draft: {topk_draft}')
print(f'hit_cnt: {hit_cnt}')
if isinstance(topk_draft, int):
    print(f'Hit rate: {hit_cnt/new_token*100:.2f}%')
else:
    for topk, hit_c in zip(topk_draft, hit_cnt):
        print(f'topk: {topk:2d} - Hit rate: {hit_c/new_token*100:.2f}%')
print(f'New token: {output_ids.size(-1)-input_len}')
print(model.tokenizer.decode(output_ids[0]))


# start = time.perf_counter()
# output_ids = model.base_model.generate()

# start = time.perf_counter()
# output_ids, new_token, idx, reach_leaf = model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512, log=True)
# torch.cuda.synchronize()
# end = time.perf_counter()


# print('New tokens:', new_token)
# output=model.tokenizer.decode(output_ids[0])
# print('Rounds:', idx+1)
# print('Reach leaf:', reach_leaf)

# print('\n=========OUTPUT=========')
# print(output)
# print(f'Total Inference time: {end - start:.2f}s')