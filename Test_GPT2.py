from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from torch.utils.tensorboard import SummaryWriter
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(tokenizer)
config = GPT2Config.from_pretrained('gpt2')
print(config)
model = GPT2Model.from_pretrained('gpt2')
print(model)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input.data)
print(type(encoded_input))

input_ids = tokenizer.convert_tokens_to_ids(text)

res = model.generate(input_ids=input_ids, max_length=100, do_sample=False, use_cache=True)

# dict_input = {}
# for k in encoded_input.keys():
#     dict_input[k] = encoded_input[k]
#
#
#
# # sh: tensorboard --logdir [the log directory]
# # root = 'C:/Users/SUST/Desktop/'
# dir = 'runs/my_environment'
#
# with SummaryWriter(log_dir=dir, comment='YourModel') as w:
#     w.add_graph(model, dict_input, use_strict_trace=False)

# output = model(**encoded_input)
# last_hidden_state = output.last_hidden_state
# past_key_values = output.past_key_values
# hidden_states = output.hidden_states
# attentions = output.attentions
# print(type(last_hidden_state), type(past_key_values), type(hidden_states), type(attentions))
# print(last_hidden_state.shape)