# extract all model weight and test local inference, compare the result to that of model(*)

import torch
import torch.nn as nn
from autoregressive_inference import *

config, tokenizer, model = load_local_pretrained_model(model_path)

transformer_model = model.transformer

# 根据input_ids或者input_pos查表获得embedding vector
token_embedding_weight = transformer_model.wte.weight
position_embedding_weight = transformer_model.wpe.weight

transformer_layers = transformer_model.h
layers_weight = []
# todo: 确保直接通过以下权重inference的结果能够等于直接调用model.forward(*)的结果
for transformer_layer in transformer_layers:
    # Multi-head Attention block
    MHA = transformer_layer.attn
    # Conv1D: X * (W_Q|W_K|W_V) = (Q|K|V)
    attn_proj_weight, attn_proj_bias = MHA.c_attn.weight, MHA.c_attn.bias
    # Conv1D
    attn_Wo_weight, attn_Wo_bias = MHA.c_proj.weight, MHA.c_proj.bias
    # LayerNorm of MHA
    ln1 = transformer_layer.ln_1
    ln1_weight, ln1_bias = ln1.weight, ln1.bias

    # MLP block
    MLP = transformer_layer.mlp
    # Conv1D: the first linear layer
    fc_weight, fc_bias = MLP.c_fc.weight, MLP.c_fc.bias
    # Conv1D: the second linear layer
    proj_weight, proj_bias = MLP.c_proj.weight, MLP.c_proj.bias
    # LayerNorm of MLP
    ln2 = transformer_layer.ln_2
    ln2_weight, ln2_bias = ln2.weight, ln2.bias



