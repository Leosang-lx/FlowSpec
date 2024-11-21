import time
import torch.nn as nn
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoConfig
# from transformers import AutoModel, BertTokenizer, GPT2Config
from sampling import apply_sampling
import os


def check_files_exist(model_directory):
    required_files = {
        'config.json': False,
        'pytorch_model.bin': False,  # 或者是 model.bin 或其他模型权重文件名
        # 'special_tokens_map.json': False,
        # 'tokenizer_config.json': False,
        'vocab.txt': False,  # 或者其他词汇表文件名
    }
    # 检查所有必要的文件是否都存在
    for filename in required_files.keys():
        filepath = os.path.join(model_directory, filename)
        if os.path.exists(filepath):
            required_files[filename] = True
        else:
            print(f"Missing required file: {filepath}")

    return all(required_files.values())


def load_local_pretrained_model(model_dir, what=None):
    pretrained = ()
    if what is None:
        what = ['config', 'tokenizer', 'model']
    if 'config' == what or 'config' in what:
        config = AutoConfig.from_pretrained(model_dir)
        pretrained = pretrained + (config,)
    if 'tokenizer' == what or 'tokenizer' in what:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        pretrained = pretrained + (tokenizer,)
    if 'model' == what or 'model' in what:
        model = GPT2LMHeadModel.from_pretrained(model_dir)  # automodel only output the last_hidden_state
        pretrained = pretrained + (model,)

    return pretrained


def prefill(model: nn.Module, input_ids: torch.Tensor, use_cache=True, do_sample=False, **kwargs):
    """
    Generate the first token together with KV cache
    :param model:
    :param input_ids:
    :param use_cache:
    :param do_sample:
    :param kwargs:
    :return:
    """
    # local
    if len(input_ids) > max_length:
        raise Exception("Input is too long")

    with torch.no_grad():
        outputs = model(input_ids, use_cache=use_cache)

    # output logits: (batch, sequence, vocab_size)
    logits = outputs.logits
    logits_first_generated_token = logits[:, -1, :]  # (batch, vocab_size)
    first_generated_token = logits2token(logits_first_generated_token, do_sample=do_sample, **kwargs)
    past_key_values = outputs.past_key_values
    return first_generated_token, past_key_values


def logits2token(logits: torch.Tensor, do_sample=False, **kwargs):
    if do_sample:
        new_token = apply_sampling(logits, **kwargs)
    else:
        new_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
    return new_token


def decode(model: nn.Module, input_ids: torch.Tensor, max_length, use_cache=True, past_key_values=None, do_sample=False):
    """
    Is use_cache: based on the first generated token and KV cache, continuously generate the following tokens
    Not use_cache: inference with the additional token sequence
    """
    batch_size, input_length = input_ids.shape

    if input_length >= max_length:
        raise Exception("Input is too long")

    generated_text = ""
    if use_cache:
        input_ids = input_ids[..., -1:]

    with torch.no_grad():
        for _ in tqdm(range(input_length, max_length)):

            # forward with necessary input tokens
            outputs = model(input_ids, use_cache=use_cache, past_key_values=past_key_values)

            # output logits: (batch, sequence, vocab_size)
            logits = outputs.logits
            logits_new_token = logits[:, -1, :]  # (batch, vocab_size)

            # generate the next token based on the predicted logits
            next_token = logits2token(logits_new_token, do_sample=do_sample)

            # input_ids: for the i^th round of token generation, it only input with the latest token and past KV-cache
            if use_cache:
                # only require the last token for decoding phase
                input_ids = next_token
                # obtain past key_values from model outputs
                past_key_values = outputs.past_key_values
            else:
                # concat the generated token to the existing sequence
                input_ids = torch.concat((input_ids, next_token), dim=-1)

            # add the new token to the text sequence
            if batch_size == 1:
                new_token = tokenizer.convert_ids_to_tokens(next_token)  # skip_special_tokens=True when necessary
                # print(new_tokens)
                generated_text += new_token[-1]

                # stop generation when meeting the [EOS] token
                if tokenizer.eos_token_id == int(next_token[0, 0]):
                    break
    return generated_text


def autoregressive_inference(model: nn.Module, input_text: str, max_length, use_cache=True, do_sample=False, **kwargs):
    print(input_text)
    input_ids = torch.LongTensor([tokenizer.convert_tokens_to_ids(list(input_text))]).to(device)

    first_token, past_key_values = prefill(model, input_ids, use_cache=use_cache, do_sample=do_sample, **kwargs)
    # if use_cache:
    #     input_ids = first_token
    # else:
    input_ids = torch.concat((input_ids, first_token), dim=-1)

    generated_text = decode(model, input_ids, max_length, use_cache=use_cache, past_key_values=past_key_values, do_sample=do_sample)
    print(generated_text)


def get_model_path(cache_path, model_tag):
    developer, model_name = tuple(model_tag.split('/'))
    model_dir = f'models--{developer}--{model_name}'
    snapshots_dir = os.path.join(cache_path, model_dir, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if len(snapshots) > 0:
            return os.path.join(snapshots_dir, snapshots[0])
        else:
            raise Exception('No cache found in directory "snapshots"')
    else:
        return None


# load pre-trained model, tokenizer and configuration of GPT-2
# project_dir = os.getcwd()
# print(project_dir)

# model_tag = "uer/gpt2-chinese-cluecorpussmall"  # GPT2-small
# model_tag = 'uer/gpt2-large-chinese-cluecorpussmall'  # GPT2-large
model_tag = 'uer/gpt2-xlarge-chinese-cluecorpussmall'  # GPT2-xlarge

developer_name, model_name = tuple(model_tag.split('/'))
cache_path = "model_file"

model_path = get_model_path(cache_path, model_tag)

# model_path = pathlib.Path(f'{cache_path}/models--{developer_name}--{model_name}/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3')
# model_path = project_dir.joinpath(model_path)




# # config = GPT2Config.from_pretrained(model_path)
# config = AutoConfig.from_pretrained(model_path)
# # print(config)
#
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device = 'cpu'
# print(f'Device={device}')
#
# model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
# # model = AutoModel.from_pretrained(model_path).to(device)
# model.eval()
#
# # tokenizer = BertTokenizer.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

if __name__ == '__main__':

    # check necessary files
    if check_files_exist(model_path):
        print("All required files are present.")
    else:
        print("Some required files are missing.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Device={device}')

    config, tokenizer, model = load_local_pretrained_model(model_path)
    model = model.to(device)
    model_config = (config.vocab_size, config.n_positions, config.n_layer, config.n_embd, config.n_head,
                    config.n_embd // config.n_head, 4)

    # 输入文本
    # 101 tokens
    text = "在一个风和日丽的下午，小镇的街道上人来人往，孩子们在巷口追逐嬉戏。李阿姨拿着刚从市场买回来的菜篮子，步履轻盈地走回家。街边的老槐树下，几位老人正围坐在一起下象棋，不时传来欢声笑语。今天是不是一个好日子？"
    # 5 tokens
    # text = "明天降温了"

    print('Input length:', len(text))
    # token generation
    generated_text = text
    max_length = 200  # limit length
    print('Max length:', max_length)

    batch_size = 1
    input_ids = torch.LongTensor([tokenizer.convert_tokens_to_ids(list(text))]).to(device)
    # input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')  # 将文本编码为ID

    # KV-cache for the inference request
    past_key_values = None
    use_cache = True
    do_sample = False
    top_k = 20
    top_p = 0.6
    print(f'use_kv_cache={use_cache}')
    print(f'do_sample={do_sample}')

    autoregressive_inference(model, text, max_length, use_cache, do_sample)
