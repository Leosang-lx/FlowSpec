# extract all model weight and test local inference, compare the result to that of model(*)
import torch
import torch.nn.functional as F
import transformers.activations

from autoregressive_inference import *


def get_transformer_model_weight(transformer_model):
    # 根据input_ids或者input_pos查表获得embedding vector
    token_embedding_weight = transformer_model.wte.weight
    position_embedding_weight = transformer_model.wpe.weight

    # transformer_layers = transformer_model.h
    layers_weight = []
    # KV_cache = []

    # Temporary for GPT-2
    for transformer_layer in transformer_model.h:
        # Multi-head Attention block
        MHA = transformer_layer.attn
        # Conv1D: X * (W_Q|W_K|W_V) = (Q|K|V)
        attn_proj_w_b = MHA.c_attn.weight, MHA.c_attn.bias
        # Conv1D: W_o
        attn_Wo_w_b = MHA.c_proj.weight, MHA.c_proj.bias
        # LayerNorm of MHA
        ln1 = transformer_layer.ln_1
        ln1_w_b = ln1.weight, ln1.bias

        # MLP block
        MLP = transformer_layer.mlp
        # Conv1D: the first linear layer
        mlp1_w_b = MLP.c_fc.weight, MLP.c_fc.bias
        # Conv1D: the second linear layer
        mlp2_w_b = MLP.c_proj.weight, MLP.c_proj.bias
        # LayerNorm of MLP
        ln2 = transformer_layer.ln_2
        ln2_w_b = ln2.weight, ln2.bias

        layer_weight = {
            'MHA': (attn_proj_w_b, attn_Wo_w_b),
            'MLP': (mlp1_w_b, mlp2_w_b),
            'LN': (ln1_w_b, ln2_w_b)
        }
        layers_weight.append(layer_weight)

    # no bias
    # lm_head_weight = model.lm_head

    # extracted model weights
    model_weight = {
        'embedding_weights': (token_embedding_weight, position_embedding_weight),
        'layers_weights': layers_weight,
        # 'lm_head_weights': lm_head_weight
    }

    return model_weight


def split_heads(x, h, d_h):
    new_shape = x.size()[:-1] + (h, d_h)
    x = x.view(new_shape)
    return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


def merge_heads(x, h, d_h):
    x = x.permute(0, 2, 1, 3).contiguous()
    new_shape = x.size()[:-2] + (h * d_h,)
    return x.view(new_shape)


# Conv1D in GPT-2, can be replaced by nn.Linear
def forward_Conv1D_use_weights(x, weights, output_features):
    size_out = x.size()[:-1] + (output_features,)
    weight, bias = weights
    x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
    x = x.view(size_out)
    return x


# only self attention
def attn(Q, K, V, attention_mask=None, head_mask=None):
    attn_weights = torch.matmul(Q, K.transpose(-1, -2))
    # scale
    attn_weights = attn_weights / torch.full([], V.size(-1) ** 0.5, dtype=attn_weights.dtype,
                                             device=attn_weights.device)

    # causal mask: transformers -> models -> gpt2 -> modeling_gpt2.py -> GPT2Attention -> self._attn
    query_length, key_length = Q.size(-2), K.size(-2)
    causal_mask = causal_mask_cache[:, :, key_length - query_length: key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    # mixed-precision
    # if attn_weights.dtype != V.dtype:
    #     attn_weights = attn_weights.type(V.dtype)

    # head weights
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, V)

    return attn_output, attn_weights


def MHA_forward_use_weights(hidden_states, MHA_weights, layer_cache=None):
    attn_proj_w_b, attn_Wo_w_b = MHA_weights
    # QKV projection
    attn_proj_w, attn_proj_b = attn_proj_w_b
    QKV = forward_Conv1D_use_weights(hidden_states, attn_proj_w_b, 3 * hidden_states.size(-1))
    # size_out = hidden_states.size()[:-1] + (3 * hidden_states.size(-1),)  # d_model = hidden_states.size(-1)
    # QKV = torch.addmm(attn_proj_b, hidden_states.view(-1, hidden_states.size(-1)), attn_proj_w)
    # QKV = QKV.view(size_out)

    Q, K, V = QKV.split(d_model, dim=2)
    Q = split_heads(Q, h, d_h)
    K = split_heads(K, h, d_h)
    V = split_heads(V, h, d_h)

    if layer_cache is not None:
        # update layer_cache
        K_cache, V_cache = layer_cache
        K = torch.cat((K_cache, K), dim=2)
        V = torch.cat((V, V_cache), dim=2)

    layer_cache_present = (K, V) if layer_cache else None

    attn_output, attn_weights = attn(Q, K, V)

    attn_output = merge_heads(attn_output, h, d_h)
    # Wo
    attn_Wo_w, attn_Wo_b = attn_Wo_w_b
    attn_output = forward_Conv1D_use_weights(attn_output, attn_Wo_w_b, d_model)
    # size_out = hidden_states.size()[:-1] + (d_model,)
    # attn_output = torch.addmm(attn_Wo_b, attn_output.view(-1, attn_output.size(-1)), attn_Wo_w)
    # attn_output = attn_output.view(size_out)
    attn_output = F.dropout(attn_output, dropout_prob, False, False)

    return attn_output, layer_cache_present


def MLP_forward_use_weights(hidden_states, MLP_weights):
    mlp1_w_b, mlp2_w_b = MLP_weights
    mlp1_w, mlp1_b = mlp1_w_b
    mlp2_w, mlp2_b = mlp2_w_b
    # mlp1
    hidden_states = forward_Conv1D_use_weights(hidden_states, mlp1_w_b, rate * d_model)
    # size_out = hidden_states.size()[:-1] + (rate * d_model,)
    # hidden_states = torch.addmm(mlp1_b, hidden_states.view(-1, hidden_states.size(-1)), mlp1_w)
    # hidden_states = hidden_states.view(size_out)
    # activation
    # 暂时只考虑了GPT-2用的new gelu
    hidden_states = transformers.activations.NewGELUActivation().forward(hidden_states)
    # hidden_states = act(hidden_states)
    # mlp2
    hidden_states = forward_Conv1D_use_weights(hidden_states, mlp2_w_b, d_model)
    # size_out = hidden_states.size()[:-1] + (d_model,)
    # hidden_states = torch.addmm(mlp2_b, hidden_states.view(-1, hidden_states.size(-1)), mlp2_w)
    # hidden_states = hidden_states.view(size_out)
    # dropout
    hidden_states = F.dropout(hidden_states, dropout_prob, False, False)

    return hidden_states


def forward_layer_use_weights(hidden_states, layer_weights, layer_cache=None):
    attn_proj_w_b, attn_Wo_w_b = layer_weights['MHA']
    mlp1_w_b, mlp2_w_b = layer_weights['MLP']
    ln1_w_b, ln2_w_b = layer_weights['LN']

    residual = hidden_states
    # LN1
    hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, layer_norm_eps)
    # MHA
    attn_output, layer_cache_present = MHA_forward_use_weights(hidden_states, (attn_proj_w_b, attn_Wo_w_b),
                                                               layer_cache)
    # residual connection
    hidden_states = residual + attn_output

    residual = hidden_states
    # LN2
    hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, layer_norm_eps)
    # MLP
    hidden_states = MLP_forward_use_weights(hidden_states, (mlp1_w_b, mlp2_w_b))
    hidden_states = residual + hidden_states

    return hidden_states


# output use model weights
def forward_use_weights(input_ids, model_weights, KV_cache=None, device='cpu'):
    # token_embedding_weight, position_embedding_weight = model_weights['embedding_weights']
    layers_weight = model_weights['layers_weights']
    # lm_head_weight = model_weights['lm_head_weights']

    token_embedding = transformer_model.wte(input_ids)
    if KV_cache is None:
        past_length = 0
        KV_cache = tuple([None] * n_layer)
    else:
        past_length = KV_cache[0][0].size(-2)
    position_ids = torch.arange(past_length, input_ids.shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)
    position_embedding = transformer_model.wpe(position_ids)
    hidden_states = token_embedding + position_embedding

    for layer_weight, layer_cache in zip(layers_weight, KV_cache):
        hidden_states = forward_layer_use_weights(hidden_states, layer_weight, layer_cache)
        # attn_proj_w_b, attn_Wo_w_b = layer_weight['MHA']
        # mlp1_w_b, mlp2_w_b = layer_weight['MLP']
        # ln1_w_b, ln2_w_b = layer_weight['LN']
        #
        # residual = hidden_states
        # # LN1
        # hidden_states = F.layer_norm(hidden_states, (d_model,), *ln1_w_b, layer_norm_eps)
        # # MHA
        # attn_output, layer_cache_present = MHA_forward_use_weights(hidden_states, (attn_proj_w_b, attn_Wo_w_b),
        #                                                            layer_cache)
        #
        # # residual connection
        # hidden_states = residual + attn_output
        #
        # residual = hidden_states
        # # LN2
        # hidden_states = F.layer_norm(hidden_states, (d_model,), *ln2_w_b, layer_norm_eps)
        # # MLP
        # hidden_states = MLP_forward_use_weights(hidden_states, (mlp1_w_b, mlp2_w_b))
        #
        # # residual connection
        # hidden_states = residual + hidden_states
    hidden_states = transformer_model.ln_f(hidden_states)

    return hidden_states


if __name__ == '__main__':
    config, tokenizer, model = load_local_pretrained_model(model_path)
    model.eval()
    print(f'model.training={model.training}')
    torch.manual_seed(100)

    transformer_model = model.transformer

    model_weight = get_transformer_model_weight(transformer_model)

    # config
    input_length = 100
    vocab_size, max_p, n_layer, d_model, h, d_h, rate = model_config = \
        (
            config.vocab_size, config.n_positions, config.n_layer, config.n_embd, config.n_head,
            config.n_embd // config.n_head,
            4)
    layer_norm_eps = config.layer_norm_epsilon
    # dropout_prob = config.resid_pdrop
    dropout_prob = 0

    # causal mask for attn
    causal_mask_cache = torch.tril(torch.ones((max_p, max_p), dtype=torch.bool)).view(1, 1, max_p, max_p)

    casual_input = torch.randn(1, 100, d_model)

    # # test LN: pass
    # ln1_1 = transformer_model.h[0].ln_1
    # output = ln1_1(casual_input)
    # ln1_1_weights = model_weight['layers_weights'][0]['LN'][0]
    # output2 = F.layer_norm(casual_input, (d_model,), *ln1_1_weights, eps=layer_norm_eps)
    # print(torch.equal(output, output2))

    # test attn: pass
    # attn1 = transformer_model.h[0].attn
    # output = attn1(casual_input)[0]
    # attn1_weights = model_weight['layers_weights'][0]['MHA']
    # output2 = MHA_forward_use_weights(casual_input, attn1_weights)[0]
    # print(torch.allclose(output, output2))

    # test layer input: pass
    first_layer = transformer_model.h[0]
    first_layer_output = first_layer(casual_input)[0]
    first_layer_weight = model_weight['layers_weights'][0]
    my_first_layer_output = forward_layer_use_weights(casual_input, first_layer_weight)
    print(torch.equal(first_layer_output, my_first_layer_output))

    text = "在一个风和日丽的下午，小镇的街道上人来人往，孩子们在巷口追逐嬉戏。李阿姨拿着刚从市场买回来的菜篮子，步履轻盈地走回家。街边的老槐树下，几位老人正围坐在一起下象棋，不时传来欢声笑语。今天是不是一个好日子？"
    # try inference with weights only
    test_input = torch.LongTensor([tokenizer.convert_tokens_to_ids(list(text))])  # .to(device)

    # target_output
    with torch.no_grad():
        output_logits = model(test_input).logits

    hidden_states = forward_use_weights(test_input, model_weight)
    logits = model.lm_head(hidden_states)

    assert logits.shape == output_logits.shape
    print(torch.allclose(logits, output_logits))
