import torch

from autoregressive_inference import *
from simple_test.split_layer_norm import split_LN
from forward_use_weight import *


def split_weight_TP(model_weights, split_nums: int | list[int], config, split_embedding=False, split_MLP=True):
    """
    split weights for TP, only has the weights in **layers**
    :param config: hyper params of the transformer model: configuration
    :param split_embedding: False: normal TP; True: split embeddings
    :param model_weights: complete weight of a transformer model for inference
    :param heads:
    :param split_nums: int or list(int), the heads number in each partition
    :return:
    """

    assert not (split_embedding and not split_MLP)
    h = config.h
    # if isinstance(split_nums, int):  # equal split
    #     assert h % split_nums == 0
    # else:  # unbalanced split
    #     assert sum(split_nums) == h
    # # keep complete embedding weight
    # embedding_weights = model_weights['embedding_weights']

    layers_weights = model_weights['layers_weights']
    if isinstance(split_nums, int):  # equal split
        # split_cnt: how many partitions
        split_cnt = split_nums
        # split_nums: how many heads in each partition (equal)
        split_nums = [h // split_cnt for _ in range(split_cnt)]

    else:  # unbalanced split
        split_cnt = len(split_nums)

    split_layers_weights = [[] for _ in range(split_cnt)]
    for layer_idx, layer_weights in enumerate(layers_weights):
        # MHA block
        MHA_weights = layer_weights['MHA']
        attn_proj_w_b, attn_Wo_w_b = MHA_weights
        attn_proj_w, attn_proj_b = attn_proj_w_b

        split_embedding_num = [sn * config.d_h for sn in split_nums]

        # split QKV projection weights todo: detach() and clone() for torch.tensor
        split_QKV_proj = [[] for _ in range(split_cnt)]
        # split into projection of Q, K, V
        attn_proj_w_QKV = attn_proj_w.split(config.d_model, dim=-1)
        attn_proj_b_QKV = attn_proj_b.split(config.d_model, dim=-1)

        for attn_proj_w_, attn_proj_b_ in zip(attn_proj_w_QKV, attn_proj_b_QKV):  # Q_proj, K_proj, V_proj
            split_attn_proj_ws_ = attn_proj_w_.split(split_embedding_num, dim=-1)
            split_attn_proj_ws_ = [partition.clone() for partition in split_attn_proj_ws_]  # clone()
            split_attn_proj_bs_ = attn_proj_b_.split(split_embedding_num, dim=-1)
            split_attn_proj_bs_ = [partition.clone() for partition in split_attn_proj_bs_]  # clone()

            for i, split_attn_proj_wb_ in enumerate(zip(split_attn_proj_ws_, split_attn_proj_bs_)):  # split_cnt
                split_QKV_proj[i].append(split_attn_proj_wb_)

        # split multi-head projection weights Wo
        attn_Wo_w, attn_Wo_b = attn_Wo_w_b
        split_attn_Wo_ws = attn_Wo_w.split(split_embedding_num, dim=0)
        split_attn_Wo_ws = [partition.clone() for partition in split_attn_Wo_ws]  # clone()
        if split_embedding:  # split_embedding: split bias as well
            split_attn_Wo_bs = attn_Wo_b.split(split_embedding_num, dim=-1)
            split_attn_Wo_bs = [partition.clone() for partition in split_attn_Wo_bs]  # clone()
        else:  # normal TP: let rank0 device keep the whole bias
            split_attn_Wo_bs = (attn_Wo_b,) + (None,) * (split_cnt - 1)
        split_Wo_proj = tuple(zip(split_attn_Wo_ws, split_attn_Wo_bs))

        # MLP block
        MLP_weights = layer_weights['MLP']
        mlp1_wb, mlp2_wb = MLP_weights

        if not split_MLP:
            split_mlp1_wb = [mlp1_wb] * split_cnt
            split_mlp2_wb = [mlp2_wb] * split_cnt

        else:
            # split mlp1 weights
            mlp1_w, mlp1_b = mlp1_wb
            split_lens = [sen * config.rate for sen in
                          split_embedding_num]  # todo: MLP may need another split setting, share the attn split setting first
            split_mlp1_ws = mlp1_w.split(split_lens, dim=-1)
            split_mlp1_ws = [partition.clone() for partition in split_mlp1_ws]  # clone()
            split_mlp1_bs = mlp1_b.split(split_lens, dim=-1)
            split_mlp1_bs = [partition.clone() for partition in split_mlp1_bs]  # clone()
            split_mlp1_wb = tuple(zip(split_mlp1_ws, split_mlp1_bs))

            # split_mlp2 weights
            mlp2_w, mlp2_b = mlp2_wb
            split_mlp2_ws = mlp2_w.split(split_lens, dim=0)
            split_mlp2_ws = [partition.clone() for partition in split_mlp2_ws]  # clone()
            if split_embedding and layer_idx < config.n_layer - 1:  # split embedding as well
                split_mlp2_bs = mlp2_b.split(split_embedding_num, dim=-1)
                split_mlp2_bs = [partition.clone() for partition in split_mlp2_bs]  # clone()
            else:  # let rank0 device keep the whole bias
                # mlp2_b is applied for once after the results of ReduceSum, which is applied by only one partition
                split_mlp2_bs = (mlp2_b,) + (None,) * (split_cnt - 1)
            split_mlp2_wb = tuple(zip(split_mlp2_ws, split_mlp2_bs))

        layer_ln_weights = layer_weights['LN']

        if split_embedding:  # split LayerNorm
            ln1_w_b, ln2_w_b = layer_ln_weights
            # split LN1
            ln1_w, ln1_b = ln1_w_b
            if layer_idx == 0:
                split_ln1_ws = [ln1_w] * split_cnt
                split_ln1_bs = [ln1_b] * split_cnt
            else:
                split_ln1_ws = ln1_w.split(split_embedding_num, dim=-1)
                split_ln1_ws = [partition.clone() for partition in split_ln1_ws]
                split_ln1_bs = ln1_b.split(split_embedding_num, dim=-1)
                split_ln1_bs = [partition.clone() for partition in split_ln1_bs]

            # split LN2
            ln2_w, ln2_b = ln2_w_b
            split_ln2_ws = ln2_w.split(split_embedding_num, dim=-1)
            split_ln2_ws = [partition.clone() for partition in split_ln2_ws]
            split_ln2_bs = ln2_b.split(split_embedding_num, dim=-1)
            split_ln2_bs = [partition.clone() for partition in split_ln2_bs]

        for idx_split in range(split_cnt):
            split_layer_weights = {
                'MHA': (tuple(split_QKV_proj[idx_split]), split_Wo_proj[idx_split]),
                'MLP': (split_mlp1_wb[idx_split], split_mlp2_wb[idx_split])
            }
            if split_embedding:
                split_layer_weights['LN'] = (split_ln1_ws[idx_split], split_ln1_bs[idx_split]), \
                    (split_ln2_ws[idx_split], split_ln2_bs[idx_split])
            else:
                split_layer_weights['LN'] = layer_ln_weights
            split_layers_weights[idx_split].append(split_layer_weights)

    return split_layers_weights


def forward_layers_tp_local(hidden_states: torch.Tensor, split_weights_layers, config, cache, use_cache=True):
    distributed_cache, cache_present = cache
    layers_weights = model_weight['layers_weights']

    # debug
    split_cnt = len(split_weights_layers)
    split_embedding_sizes = [config.d_model // split_cnt] * split_cnt

    for layer_idx in range(n_layer):
        # check correctness of output by layer
        # original_layer = transformer_model.h[layer_idx]
        # correct_layer_output = original_layer(hidden_states)[0]
        print('=========Layer', layer_idx)

        layer_weights = layers_weights[layer_idx]
        ln1_w_b, ln2_w_b = layer_weights['LN']

        split_weights_layer = [split_weights_device[layer_idx] for split_weights_device in
                               split_weights_layers]  # 获得这层layer_idx所有的split_weights
        split_layer_cache = [split_cache_device[layer_idx] for split_cache_device in distributed_cache]

        residual = hidden_states
        # LN1
        hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, config.ln_eps)
        split_embeddings = hidden_states.split(split_embedding_sizes, dim=-1)
        print('LN1 output:', [se.reshape(-1)[:2] for se in split_embeddings])

        MHA_tp_outputs = []

        # MHA
        for device_idx, (split_weight_layer, split_cache) in enumerate(zip(split_weights_layer, split_layer_cache)):
            split_MHA_weight = split_weight_layer['MHA']
            # split_QKV_proj, split_Wo_proj = split_MHA_weight
            split_MHA_output, layer_cache = MHA_forward_use_weights(hidden_states, split_MHA_weight, config,
                                                                    split_cache)
            MHA_tp_outputs.append(split_MHA_output)
            if use_cache:
                cache_present[device_idx] = cache_present[device_idx] + (layer_cache,)

        # AllReduce
        MHA_output = sum(MHA_tp_outputs)
        split_embeddings = MHA_output.split(split_embedding_sizes, dim=-1)
        print('Wo output:', [se.reshape(-1)[:2] for se in split_embeddings])

        # residual connection
        hidden_states = residual + MHA_output
        split_embeddings = hidden_states.split(split_embedding_sizes, dim=-1)
        print('Residual1 output:', [se.reshape(-1)[:2] for se in split_embeddings])

        residual = hidden_states
        # LN2
        hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, config.ln_eps)
        split_embeddings = hidden_states.split(split_embedding_sizes, dim=-1)
        print('LN2 output:', [se.reshape(-1)[:2] for se in split_embeddings])
        # MLP
        MLP_tp_results = []
        for split_weight_layer in split_weights_layer:
            split_MLP_weight = split_weight_layer['MLP']
            split_MLP_result = MLP_forward_use_weights(hidden_states, split_MLP_weight, config)
            MLP_tp_results.append(split_MLP_result)

        # AllReduce
        MLP_output = sum(MLP_tp_results)

        split_embeddings = MLP_output.split(split_embedding_sizes, dim=-1)
        print('MLP output:', [se.reshape(-1)[:2] for se in split_embeddings])
        # residual connection
        hidden_states = residual + MLP_output
        split_embeddings = hidden_states.split(split_embedding_sizes, dim=-1)
        print('Residual2 output:', [se.reshape(-1)[:2] for se in split_embeddings])

        # when atol=1e-7, allclose() is False
        # print(f'layer {layer_idx}', torch.allclose(hidden_states, correct_layer_output, atol=1e-6))
    return hidden_states, cache_present


def reduce_scatter_overlap_local(xis, Wis, bis):
    assert len(xis) == len(Wis) == len(bis)
    split_cnt = len(xis)
    W_in, W_out = Wis[0].shape
    assert xis[0].size(-1) == W_in and W_out % split_cnt == 0

    W_split_size = W_out // split_cnt
    Wis_split = [Wi.split(W_split_size, dim=-1) for Wi in Wis]
    split_outputs = []
    for i in range(split_cnt):
        # split_output = [Conv1D_forward_use_weights(xi, Wi_split[i]) for xi, Wi_split in zip(xis, Wis_split)]
        split_output = [torch.matmul(xi, Wi_split[i]) for xi, Wi_split in zip(xis, Wis_split)]
        split_output = sum(split_output) + bis[i]
        split_outputs.append(split_output)

    return split_outputs


def gather_reduce_overlap_local(xis, Wis, bis):
    assert len(xis) == len(Wis) == len(bis)
    split_cnt = len(xis)
    W_in, W_out = Wis[0].shape
    assert W_in % split_cnt == 0 and xis[0].size(-1) == (W_in // split_cnt)
    W_split_size = W_in // split_cnt
    Wis_split = [Wi.split(W_split_size, dim=0) for Wi in Wis]
    split_outputs = []
    for i in range(split_cnt):
        # split_output = [Conv1D_forward_use_weights(xi, Wi_s) for xi, Wi_s in zip(xis, Wis_split[i])]
        split_output = [torch.matmul(xi, Wi_s) for xi, Wi_s in zip(xis, Wis_split[i])]
        split_output = sum(split_output) + bis[i]
        split_outputs.append(split_output)

    return split_outputs


def split_layer_norm_local(split_embeddings, split_ln_params, eps):
    x_mean = sum([se.sum(dim=-1, keepdim=True) for se in split_embeddings]).div_(d_model)
    x_var = sum([(se - x_mean).square().sum(dim=-1, keepdim=True) for se in split_embeddings]).div_(d_model)
    split_embeddings = [split_LN(se, x_mean, x_var, *split_ln_params[i], eps) for i, se in enumerate(split_embeddings)]
    return split_embeddings


def forward_layers_se_local(hidden_states: torch.Tensor, split_weights_layers, config, cache,
                                         use_cache=True):
    distributed_cache, cache_present = cache

    print('=========Layer', 0)
    # residual
    residual = hidden_states
    # LN1
    ln1_w_b, _ = model_weight['layers_weights'][0]['LN']
    hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, config.ln_eps)

    # forward the first MHA block: using normal TP
    # MHA except for the last Wo
    split_first_MHA_weights = [split_weights_device[0]['MHA'] for split_weights_device in split_weights_layers]
    split_first_QKV_proj_weights = [sw[0] for sw in split_first_MHA_weights]

    split_first_caches = [split_cache_device[0] for split_cache_device in distributed_cache]

    split_embeddings = []
    for device_idx, (split_first_QKV_proj_weight, split_cache) in enumerate(
            zip(split_first_QKV_proj_weights, split_first_caches)):
        if len(split_first_QKV_proj_weight) == 3:  # Q_proj_w_b, K_proj_w_b, V_proj_w_b
            split_first_QKV_proj_weight = tuple(
                [torch.concat(ws_or_bs, dim=-1) for ws_or_bs in zip(*split_first_QKV_proj_weight)])
        attn_output, layer_cache = QKV_proj_and_attn_using_weights(hidden_states, split_first_QKV_proj_weight, config,
                                                                   split_cache)
        split_embeddings.append(attn_output)
        if use_cache:
            cache_present[device_idx] = cache_present[device_idx] + (layer_cache,)

    split_embedding_sizes = [o.size(-1) for o in split_embeddings]

    # continuous n_layer-1 inference
    for layer_idx in range(n_layer - 1):
        this_layer_split_weights = [split_weights_device[layer_idx] for split_weights_device in split_weights_layers]
        next_layer_split_weights = [split_weights_device[layer_idx + 1] for split_weights_device in
                                    split_weights_layers]

        # get Wo for reduce-scatter overlap
        last_layer_split_Wo_weights = [tl['MHA'][1] for tl in this_layer_split_weights]
        Wis = [swo[0] for swo in last_layer_split_Wo_weights]
        bis = [swo[1] for swo in last_layer_split_Wo_weights]
        split_embeddings = reduce_scatter_overlap_local(split_embeddings, Wis, bis)

        print('Wo output:', [se.reshape(-1)[:2] for se in split_embeddings])

        # residual connection
        if layer_idx == 0:
            residual_split = residual.split(split_embedding_sizes, dim=-1)
        split_embeddings = [r + s for r, s in zip(residual_split, split_embeddings)]

        print('Residual1 output:', [se.reshape(-1)[:2] for se in split_embeddings])

        # residual
        residual_split = split_embeddings

        # split LN2
        this_layer_LN2_weights = [tl['LN'][1] for tl in this_layer_split_weights]
        split_embeddings = split_layer_norm_local(split_embeddings, this_layer_LN2_weights, config.ln_eps)
        # split_embeddings = [F.layer_norm(se, (split_embedding_sizes[i],), *this_layer_LN2_weights[i], config.ln_eps) for
        #                     i, se in enumerate(split_embeddings)]
        print('LN2 output:', [se.reshape(-1)[:2] for se in split_embeddings])

        # get MLP1 for gather-reduce overlap
        this_layer_split_MLP1_weights = [tl['MLP'][0] for tl in this_layer_split_weights]
        Wis = [smlp[0] for smlp in this_layer_split_MLP1_weights]
        bis = [smlp[1] for smlp in this_layer_split_MLP1_weights]
        split_embeddings = gather_reduce_overlap_local(split_embeddings, Wis, bis)
        # split activation
        split_embeddings = [transformers.activations.NewGELUActivation().forward(se) for se in split_embeddings]
        # get MLP2 for overlap reduce-scatter overlap
        this_layer_split_MLP2_weights = [tl['MLP'][1] for tl in this_layer_split_weights]
        Wis = [smlp[0] for smlp in this_layer_split_MLP2_weights]
        bis = [smlp[1] for smlp in this_layer_split_MLP2_weights]
        split_embeddings = reduce_scatter_overlap_local(split_embeddings, Wis, bis)

        print('MLP output:', [se.reshape(-1)[:2] for se in split_embeddings])

        # residual connection
        split_embeddings = [r + s for r, s in zip(residual_split, split_embeddings)]

        print('Residual2 output:', [se.reshape(-1)[:2] for se in split_embeddings])

        print('=========Layer', layer_idx + 1)

        # residual
        residual_split = split_embeddings

        # split LN1
        next_layer_LN1_weights = [nl['LN'][0] for nl in next_layer_split_weights]
        split_embeddings = split_layer_norm_local(split_embeddings, next_layer_LN1_weights, config.ln_eps)
        # x_mean = sum([se.sum(dim=-1, keepdim=True) for se in split_embeddings]).div_(d_model)
        # x_var = sum([(se - x_mean).square().sum(dim=-1, keepdim=True) for se in split_embeddings]).div_(d_model)
        # split_embeddings = [split_LN(se, x_mean, x_var, *next_layer_LN1_weights[i], config.ln_eps) for
        #                     i, se in enumerate(split_embeddings)]
        # split_embeddings = [F.layer_norm(se, (split_embedding_sizes[i],), *next_layer_LN1_weights[i], config.ln_eps) for
        #                     i, se in enumerate(split_embeddings)]
        print('LN1 output:', [se.reshape(-1)[:2] for se in split_embeddings])

        # split QKY proj
        next_layer_QKV_proj_weights = [nl['MHA'][0] for nl in next_layer_split_weights]
        for i, nw in enumerate(next_layer_QKV_proj_weights):
            if len(nw) == 3:
                next_layer_QKV_proj_weights[i] = tuple([torch.concat(ws_or_bs, dim=-1) for ws_or_bs in zip(*nw)])
        Wis = [qkv_wb[0] for qkv_wb in next_layer_QKV_proj_weights]
        bis = [qkv_wb[1] for qkv_wb in next_layer_QKV_proj_weights]

        split_embeddings = gather_reduce_overlap_local(split_embeddings, Wis, bis)
        # attn
        split_next_layer_caches = [split_cache_device[layer_idx + 1] for split_cache_device in distributed_cache]
        for i, qkv in enumerate(split_embeddings):
            Q, K, V = split_QKV(qkv, config.d_h)
            next_layer_cache = split_next_layer_caches[i]
            if next_layer_cache is not None:
                K_cache, V_cache = next_layer_cache
                K = torch.cat((K_cache, K), dim=2)
                V = torch.cat((V_cache, V), dim=2)
            layer_cache = K, V

            attn_output, _ = attn(Q, K, V)
            split_embeddings[i] = merge_heads(attn_output, config.d_h)
            if use_cache:
                cache_present[i] = cache_present[i] + (layer_cache,)

    # the rest of the last layer: Wo, LN2, MLP1, act, MLP2
    last_layer_split_weight = [split_weights_layer[-1] for split_weights_layer in split_weights_layers]
    # get Wo for reduce-scatter overlap
    last_layer_split_Wo_weights = [ll['MHA'][1] for ll in last_layer_split_weight]
    Wis = [swo[0] for swo in last_layer_split_Wo_weights]
    bis = [swo[1] for swo in last_layer_split_Wo_weights]
    split_embeddings = reduce_scatter_overlap_local(split_embeddings, Wis, bis)

    # residual connection
    split_embeddings = [r + s for r, s in zip(residual_split, split_embeddings)]
    # residual
    residual_split = split_embeddings

    # split LN2
    this_layer_LN2_weights = [ll['LN'][1] for ll in last_layer_split_weight]
    split_embeddings = split_layer_norm_local(split_embeddings, this_layer_LN2_weights, config.ln_eps)
    # split_embeddings = [F.layer_norm(se, (split_embedding_sizes[i],), *this_layer_LN2_weights[i], config.ln_eps) for
    #                     i, se in enumerate(split_embeddings)]
    print('LN2 output:', [se.reshape(-1)[:2] for se in split_embeddings])

    # get MLP1 for gather-reduce overlap
    this_layer_split_MLP1_weights = [ll['MLP'][0] for ll in last_layer_split_weight]
    Wis = [smlp[0] for smlp in this_layer_split_MLP1_weights]
    bis = [smlp[1] for smlp in this_layer_split_MLP1_weights]
    split_embeddings = gather_reduce_overlap_local(split_embeddings, Wis, bis)
    # split activation
    split_embeddings = [transformers.activations.NewGELUActivation().forward(se) for se in split_embeddings]
    # final MLP2
    this_layer_split_MLP2_weights = [ll['MLP'][1] for ll in last_layer_split_weight]
    split_outputs = [Conv1D_forward_use_weights(se, wb) for se, wb in
                     zip(split_embeddings, this_layer_split_MLP2_weights)]
    hidden_states = sum(split_outputs)

    # residual connection
    residual = torch.concat(residual_split, dim=-1)
    hidden_states = residual + hidden_states

    return hidden_states, cache_present


# test TP local
def tp_local(input_ids, split_nums: int | list[int], split_weights_tp, config, distributed_cache=None, use_cache=True,
             split_embedding=False):
    """
    local test tensor parallelism: for result verification
    :param input_ids:
    :param split_nums: int -> number of partitions; list[int] -> number of heads in each partition
    :param split_weights_tp: split_weights by layer
    :param KV_cache: split KVCache on heads
    :return:
    """
    if isinstance(split_nums, int):
        assert config.h % split_nums == 0
        split_cnt = split_nums
        split_nums = [config.h // split_nums] * split_cnt
    else:
        assert sum(split_nums) == config.h
        split_cnt = len(split_nums)

    # centralize: embedding{
    token_embedding = transformer_model.wte(input_ids)
    if distributed_cache is None:
        past_length = 0
        distributed_cache = [[
                                 None] * n_layer] * split_num  # distributed KVCache for TP: KV_cache[device_num][layer_idx]
    else:
        past_length = distributed_cache[0][0][0].size(-2)
    if use_cache:
        cache_present = [()] * split_cnt
    else:
        cache_present = None

    position_ids = torch.arange(past_length, input_ids.shape[-1] + past_length, dtype=torch.long,
                                device=input_ids.device)
    position_ids = position_ids.unsqueeze(0)
    position_embedding = transformer_model.wpe(position_ids)
    hidden_states = token_embedding + position_embedding
    # }
    # prepare split weights

    cache = distributed_cache, cache_present
    # forward layers
    if split_embedding:
        hidden_states, cache_present = forward_layers_se_local(hidden_states, split_weights_tp, config, cache,
                                                               use_cache=True)
    else:
        hidden_states, cache_present = forward_layers_tp_local(hidden_states, split_weights_tp, config, cache,
                                                               use_cache=True)

    hidden_states = transformer_model.ln_f(hidden_states)

    return hidden_states, cache_present


if __name__ == '__main__':
    # split model weight for tensor parallelism
    config, tokenizer, model = load_local_pretrained_model(model_path)
    model.eval()
    torch.manual_seed(10)
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

    split_embedding = False
    # prefill{
    split_num = 2
    split_weights = split_weight_TP(model_weight, split_num, model_config, split_embedding=split_embedding)
    output_tp, KV_cache_tp = tp_local(input_ids, split_num, split_weights, model_config, None,
                                      split_embedding=split_embedding)
    # hidden_states = transformer_model.ln_f(output_tp)
    logits_tp = model.lm_head(output_tp)

    next_tokens = logits2token(logits_tp[:, -1, :], False)
    next_word = tokenizer.convert_ids_to_tokens(next_tokens)
    print(next_word)
    # }

    # # decoding{
    # input_ids = torch.concat([input_ids, next_tokens], dim=-1)
    # output_tp, KV_cache_tp = tp_local(input_ids, split_num, split_weights, config, KV_cache_tp)
    # logits_tp = model.lm_head(output_tp)
    #
    # from sampling import apply_sampling
    # next_tokens = apply_sampling(logits_tp[:, -1, :])
    # next_word = tokenizer.convert_ids_to_tokens(next_tokens)
    # print(next_word)
    # # }

    output = transformer_model(input_ids)
    KV_cache = output.past_key_values

    print(torch.allclose(output.last_hidden_state, output_tp, atol=1e-2))

    KV_cache_tp = list(zip(*KV_cache_tp))
    for i, layer_cache_tp in enumerate(KV_cache_tp):
        K_cache_layer = torch.concat([split_layer_cache[0] for split_layer_cache in layer_cache_tp], dim=1)
        V_cache_layer = torch.concat([split_layer_cache[1] for split_layer_cache in layer_cache_tp], dim=1)
        KV_cache_tp[i] = (K_cache_layer, V_cache_layer)

        K_cache_correct, V_cache_correct = KV_cache[i]

        print(f'Cache of layer {i}:', torch.allclose(K_cache_layer, K_cache_correct, atol=1e-3),
              torch.allclose(V_cache_layer, V_cache_correct, atol=1e-3))
