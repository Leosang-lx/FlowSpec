from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
from transformers import AutoConfig
import time
from eagle.model.utils import calculate_model_size_with_buffers

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
    total_token=64,
    depth=6
)

model_size = calculate_model_size_with_buffers(model)
print(f"Model size: {model_size:.2f} MB")

# model.eval()
# your_message="Hello"
# conv = get_conversation_template("vicuna")
# # conv = get_conversation_template("llama-2-chat")
# conv.append_message(conv.roles[0], your_message)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()
# input_ids=model.tokenizer([prompt]).input_ids
# input_ids = torch.as_tensor(input_ids).cuda()
# print(f"input_ids: {input_ids}")
# output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
# output=model.tokenizer.decode(output_ids[0])
# print(output)
# model = EaModel.from_pretrained(
#     base_model_path=base_model_path,
#     ea_model_path=EAGLE_model_path,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     device_map="auto",
#     total_token=-1,
#     # total_token=16,
#     # depth=2
# )

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

start = time.perf_counter()
output_ids, new_token, idx = model.eagenerate(input_ids,temperature=0,max_new_tokens=512, log=True)
torch.cuda.synchronize()
end = time.perf_counter()

print('New tokens:', new_token.item())
output=model.tokenizer.decode(output_ids[0])
print('Rounds:', idx+1)

print('\n=========OUTPUT=========')
print(output)
print(f'Total Inference time: {end - start:.2f}s')