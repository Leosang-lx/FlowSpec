import time
import torch
from transformers import BertTokenizer, GPT2LMHeadModel, AutoConfig

model_tag = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = BertTokenizer.from_pretrained(model_tag)
model = GPT2LMHeadModel.from_pretrained(model_tag).to("cuda")
model.eval()
model_config = AutoConfig.from_pretrained(model_tag)
print(model_config)
text = "明天降温了"  # original input text

# use .forward() to iteratively generate tokens
input_ids = torch.LongTensor([tokenizer.convert_tokens_to_ids(list(text))]).to("cuda")
initial_pred = model(input_ids=input_ids, past_key_values=None, use_cache=True)
print('Logits shape:', initial_pred.logits)
y_prob = torch.softmax(initial_pred.logits, dim=-1).argmax(dim=-1)
output_token = tokenizer.convert_ids_to_tokens(y_prob[0])
print(output_token)
print('past_key_value:', initial_pred.past_key_values)


# use model.generate() to generate the sequence
res = model.generate(input_ids, max_length=100, do_sample=True, use_cache=True, top_k=30, top_p=0.8)
print(res)
print(res.shape)
output = tokenizer.convert_ids_to_tokens(res[0], skip_special_tokens=True)
output = ''.join(output)
print(output)

# max_length = list(range(10, 1101, 100))
# for i in max_length:
#     res = model.generate(input_ids=input_ids, max_length=i, do_sample=False, use_cache=True)