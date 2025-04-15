from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
from transformers import AutoConfig
import time
from eagle.model.utils import *
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

@torch.no_grad()
def ea_test_truncate(
        model,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
):
    
    if is_llama3:
        stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    max_length=max_length-model.ea_layer.total_tokens-10

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        logits_processor = None
    # Avoid modifying the input_ids in-place

    padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)  # -1
    input_ids = input_ids.clone()
    model.ea_layer.reset_kv()

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

    input_len = input_ids.shape[1]
    reset_tree_mode(model)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, past_key_values, logits_processor
    )

    new_token = 0
    reach_leaf = 0
    total_truncate = 0
    for idx in range(max_length):
        model.base_model.model.tree_mask = tree_mask

        draft_tokens=draft_tokens.to(input_ids.device)

        logits, hidden_state_new, outputs = tree_decoding(
            model,
            draft_tokens,
            past_key_values,
            tree_position_ids,
            input_ids,
            retrieve_indices,
        )

        draft_tokens=torch.cat((draft_tokens,padding),dim=1)
        candidates=draft_tokens[0,retrieve_indices]

        best_candidate, accept_length, sample_p, truncate_cnt = evaluate_posterior(
            logits, candidates, logits_processor, test_truncate=True
        )
        total_truncate += truncate_cnt

        if log:
            cur_path_depth = (retrieve_indices[best_candidate, :] != -1).sum().item()
            print(f'{idx}th round, accept_len/depth: {accept_length+1}/{cur_path_depth}')
            if cur_path_depth == accept_length + 1:
                reach_leaf += 1

        input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state_new,
            sample_p
        )

        if is_llama3:
            if stop_token_id in input_ids[0, input_len:].tolist():
                break

        if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > max_length:
            break
    if not log:
        return input_ids
    else:
        return input_ids, new_token, idx, reach_leaf, total_truncate
    


start = time.perf_counter()
output_ids, new_token, idx, reach_leaf, total_truncate = ea_test_truncate(model, input_ids,temperature=0.5,max_new_tokens=512, log=True)
torch.cuda.synchronize()
end = time.perf_counter()

print('New tokens:', new_token)
output=model.tokenizer.decode(output_ids[0])
print('Rounds:', idx+1)
print('Reach leaf:', reach_leaf)
print(f'truncate_cnt: {total_truncate}')

print('\n=========OUTPUT=========')
print(output)
print(f'Total Inference time: {end - start:.2f}s')