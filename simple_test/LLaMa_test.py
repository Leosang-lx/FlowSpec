from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
from GPT2_autoregressive_inference import cache_path

# 模型名称
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 加载模型和分词器
config = LlamaConfig()
print(config)
model = LlamaForCausalLM(config)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
# config = AutoConfig.from_pretrained(model_name, cache_dir=cache_path)

# def get_model_weights(model: LlamaForCausalLM):
#     embeddings =

# print(config)