from autoregressive_inference import *
from test_model_weight import *


def split_weight_TP(model_weights, heads, split_nums: int | list[int]):
    """
    split weights for TP, only has the weights in **layers**
    :param model_weights:
    :param heads:
    :param split_nums: int or list(int), the heads number in each partition
    :return:
    """
    if isinstance(split_nums, int):
        assert heads % split_nums == 0
    else:
        assert sum(split_nums) == heads
    # keep complete embedding weight
    embedding_weights = model_weights['embedding_weights']

    layers_weights = model_weights['layers_weights']
    if isinstance(split_nums, int):  # equal split
        split_cnt = split_nums
        split_nums = [heads // split_cnt for _ in range(split_cnt)]

    else:  # unbalanced split
        split_cnt = len(split_nums)

    split_layers_weights = [[] for _ in range(split_cnt)]
    for layer_weights in layers_weights:
        # MHA block
        MHA_weights = layer_weights['MHA']
        attn_proj_w_b, attn_Wo_w_b = MHA_weights
        attn_proj_w, attn_proj_b = attn_proj_w_b

        split_embedding_num = [sn * d_h for sn in split_nums]

        # split QKV projection weights todo: detach() and clone() for torch.tensor
        split_QKV_proj = [[] for _ in range(split_cnt)]
        # split into projection of Q, K, V
        attn_proj_w_QKV = attn_proj_w.split(d_model, dim=-1)
        attn_proj_b_QKV = attn_proj_b.split(d_model, dim=-1)

        for attn_proj_w_, attn_proj_b_ in zip(attn_proj_w_QKV, attn_proj_b_QKV):  # Q_proj, K_proj, V_proj
            split_attn_proj_ws_ = attn_proj_w_.split(split_embedding_num, dim=-1)
            split_attn_proj_bs_ = attn_proj_b_.split(split_embedding_num, dim=-1)
            for i, split_attn_proj_wb_ in enumerate(zip(split_attn_proj_ws_, split_attn_proj_bs_)):  # split_cnt
                split_QKV_proj[i].append(split_attn_proj_wb_)

        # split multi-head projection weights Wo
        attn_Wo_w, attn_Wo_b = attn_Wo_w_b
        split_attn_Wo_ws = attn_Wo_w.split(split_embedding_num, dim=0)
        # split_attn_Wo_bs = attn_Wo_b.split(split_embedding_num, dim=-2)

        split_Wo_proj = tuple(zip(split_attn_Wo_ws, (attn_Wo_b,) + (None,) * (split_cnt - 1)))

        # MLP block
        MLP_weights = layer_weights['MLP']
        mlp1_wb, mlp2_wb = MLP_weights

        # split mlp1 weights
        mlp1_w, mlp1_b = mlp1_wb
        split_lens = [sen * rate for sen in split_embedding_num]  # todo: may need another split setting, share the attn split setting first
        split_mlp1_ws = mlp1_w.split(split_lens, dim=-1)
        split_mlp1_bs = mlp1_b.split(split_lens, dim=-1)
        split_mlp1_wb = tuple(zip(split_mlp1_ws, split_mlp1_bs))

        # split_mlp2 weights
        mlp2_w, mlp2_b = mlp2_wb
        split_mlp2_ws = mlp2_w.split(split_lens, dim=0)
        # mlp2_b is applied for once after the results of ReduceSum, which is applied by only one partition
        split_mlp2_bs = (mlp2_b,) + (None,) * (split_cnt - 1)
        split_mlp2_wb = tuple(zip(split_mlp2_ws, split_mlp2_bs))

        for idx_split in range(split_cnt):
            split_layer_weights = {
                'MHA': (tuple(split_QKV_proj[idx_split]), split_Wo_proj[idx_split]),
                'MLP': (split_mlp1_wb[idx_split], split_mlp2_wb[idx_split])
            }
            split_layers_weights[idx_split].append(split_layer_weights)

    return split_layers_weights


# test TP local
def tp_local(input_ids, split_nums: int | list[int], split_weights_tp, config, KV_cache=None):
    """
    :param input_ids:
    :param split_nums: int -> number of partitions; list[int] -> number of heads in each partition
    :param split_weights_tp:
    :param KV_cache:
    :return:
    """
    if isinstance(split_nums, int):
        assert config.h % split_nums == 0
        split_cnt = split_nums
        split_nums = [config.h // split_nums] * split_cnt
    else:
        assert sum(split_nums) == config.h
        split_cnt = len(split_nums)

    # centralize{
    token_embedding = transformer_model.wte(input_ids)
    if KV_cache is None:
        past_length = 0
        KV_cache = tuple([[None] * n_layer] * split_num)  # distributed KVCache for TP: KV_cache[device_num][layer_idx]
    else:
        past_length = KV_cache[0][0].size(-2)
    position_ids = torch.arange(past_length, input_ids.shape[-1] + past_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0)
    position_embedding = transformer_model.wpe(position_ids)
    hidden_states = token_embedding + position_embedding
    # }
    # prepare split weights
    # 先写没cache的
    layers_weights = model_weight['layers_weights']
    for layer_idx in range(n_layer):
        original_layer = transformer_model.h[layer_idx]
        correct_layer_output = original_layer(hidden_states)[0]

        layer_weights = layers_weights[layer_idx]
        split_weights_layer = [split_weights_device[layer_idx]  for split_weights_device in split_weights_tp] # 获得这层layer_idx所有的split_weights
        ln1_w_b, ln2_w_b = layer_weights['LN']

        residual = hidden_states
        # LN1
        hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, config.ln_eps)
        MHA_tp_results = []
        # MHA
        for split_weight_layer in split_weights_layer:
            split_MHA_weight = split_weight_layer['MHA']
            # split_QKV_proj, split_Wo_proj = split_MHA_weight
            split_MHA_result, layer_cache = MHA_forward_use_weights(hidden_states, split_MHA_weight, config)
            MHA_tp_results.append(split_MHA_result)

        # AllReduce
        MHA_output = sum(MHA_tp_results)
        # residual connection
        hidden_states = residual + MHA_output

        residual = hidden_states
        # LN2
        hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, config.ln_eps)
        # MLP
        MLP_tp_results = []
        for split_weight_layer in split_weights_layer:
            split_MLP_weight = split_weight_layer['MLP']
            split_MLP_result = MLP_forward_use_weights(hidden_states, split_MLP_weight, config)
            MLP_tp_results.append(split_MLP_result)

        # AllReduce
        MLP_output = sum(MLP_tp_results)
        # residual connection
        hidden_states = residual + MLP_output

        # when atol=1e-7, allclose() is False
        print(f'layer {layer_idx}', torch.allclose(hidden_states, correct_layer_output, atol=1e-6))

    hidden_states = transformer_model.ln_f(hidden_states)

    return hidden_states

if __name__ == '__main__':
    # split model weight for tensor parallelism
    config, tokenizer, model = load_local_pretrained_model(model_path)
    model.eval()
    vocab_size, max_p, n_layer, d_model, h, d_h, rate = \
        (
            config.vocab_size, config.n_positions, config.n_layer, config.n_embd, config.n_head,
            config.n_embd // config.n_head,
            4)
    layer_norm_eps = config.layer_norm_epsilon
    # dropout_prob = config.resid_pdrop
    dropout_prob = 0
    model_config = {
        'vocab_size': vocab_size,
        'max_position': max_p,
        'n_layer': n_layer,
        'd_model': d_model,
        'h': h,
        'd_h': d_h,
        'rate': rate,
        'ln_eps': layer_norm_eps,
        'dropout_prob': dropout_prob
    }
    model_config = SimpleNamespace(**model_config)
    model_config.split_heads = 3

    transformer_model = model.transformer
    model_weight = get_transformer_model_weight(transformer_model)

    text = "在一个风和日丽的下午，小镇的街道上人来人往，孩子们在巷口追逐嬉戏。李阿姨拿着刚从市场买回来的菜篮子，步履轻盈地走回家。街边的老槐树下，几位老人正围坐在一起下象棋，不时传来欢声笑语。今天是不是一个好日子？"
    input_ids = torch.LongTensor([tokenizer.convert_tokens_to_ids(list(text))])  # .to(device)

    split_num = 3
    split_weights = split_weight_TP(model_weight, h, split_num)
    output_tp = tp_local(input_ids, split_num, split_weights, model_config)
    # hidden_states = transformer_model.ln_f(output_tp)
    logits_tp = model.lm_head(output_tp)
    from sampling import apply_sampling
    next_token = apply_sampling(logits_tp[:, -1, :])
    next_word = tokenizer.convert_ids_to_tokens(next_token)
    print(next_word)

    output = transformer_model(input_ids).last_hidden_state

    print(torch.allclose(output, output_tp, ))





