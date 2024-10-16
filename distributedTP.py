from autoregressive_inference import *
from test_model_weight import get_transformer_model_weight

# split model weight for tensor parallelism
config, tokenizer, model = load_local_pretrained_model(model_path)
vocab_size, max_p, n_layer, d_model, h, d_h, rate = model_config = \
        (
            config.vocab_size, config.n_positions, config.n_layer, config.n_embd, config.n_head,
            config.n_embd // config.n_head,
            4)

transformer_model = model.transformer
model_weight = get_transformer_model_weight(transformer_model)
def split_weight_TP(model_weights, heads, split_nums: int | list[int]):
    """
    split weights
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
        split_attn_Wo_ws = attn_Wo_w.split(split_embedding_num, dim=-1)
        split_attn_Wo_bs = attn_Wo_b.split(split_embedding_num, dim=-1)

        split_Wo_proj = list(zip(split_attn_Wo_ws, split_attn_Wo_bs))

        # MLP block
        MLP_weights = layer_weights['MLP']
        mlp1_wb, mlp2_wb = MLP_weights

        # split mlp1 weights
        mlp1_w, mlp1_b = mlp1_wb
        split_lens = [sen * rate for sen in split_embedding_num]  # todo: may need another split setting, share the attn split setting first
        split_mlp1_ws = mlp1_w.split(split_lens, dim=-1)
        split_mlp1_bs = mlp1_b.split(split_lens, dim=-1)
        split_mlp1_wb = list(zip(split_mlp1_ws, split_mlp1_bs))

        # split_mlp2 weights
        mlp2_w, mlp2_b = mlp2_wb
        split_mlp2_ws = mlp2_w.split(split_lens, dim=0)
        # mlp2_b is applied for once after the results of ReduceSum, do not split

        for idx_split in range(split_cnt):
            split_layer_weights = {
                'MHA': (tuple(split_QKV_proj[idx_split]), split_Wo_proj[idx_split]),
                'MLP': (split_mlp1_wb, split_mlp2_ws)
            }
            split_layers_weights[idx_split].append(split_layer_weights)


        # LN_weights = layer_weights['LN']


split_weights = split_weight_TP(model_weight, h, 2)

