# A simple example to show TENSOR PARALLELISM locally
# using the transformer layer of GPT-2
import math
import torch
from layers.attention import AttentionLayer
from layers.feedforward import Swish, PositionwiseFeedForward
import time

# **Assume there is no bias in any linear layer**
# configuration of attention
seq = 4
hid = 2
h = heads = 2
seed = time.time()
# seed = 100
torch.manual_seed(int(seed))

A = torch.randn(seq, hid)  # input: embedded token

# MLA layer
attention_layer = AttentionLayer(heads=h, dims=hid)
attention_layer.eval()
MHA_output = None
# begin [TP MHA]

# weights
Wq = attention_layer.proj_q.weight
Wk = attention_layer.proj_k.weight
Wv = attention_layer.proj_v.weight
Wo = attention_layer.linear.weight
print(Wq.shape, Wk.shape, Wv.shape, Wo.shape)
# if bias:
# q_bias = layer.proj_q.bias
# k_bias = layer.proj_k.bias
# v_bias = layer.proj_v.bias
# print(q_bias.shape, k_bias.shape, v_bias.shape)

# Real QKV for value comparison with distributed style
Q = A @ Wq.T
K = A @ Wk.T
V = A @ Wv.T

# split weights
Wqh1 = Wq[0, :].unsqueeze(0)
Wqh2 = Wq[1, :].unsqueeze(0)
Wkh1 = Wk[0, :].unsqueeze(0)
Wkh2 = Wk[1, :].unsqueeze(0)
Wvh1 = Wv[0, :].unsqueeze(0)
Wvh2 = Wv[1, :].unsqueeze(0)
Woh1 = Wo[:, 0].unsqueeze(1)
Woh2 = Wo[:, 1].unsqueeze(1)

# model weights of layer, needs to transpose: need .T
Qh1, Kh1, Vh1 = A @ Wqh1.T, A @ Wkh1.T, A @ Wvh1.T
Qh2, Kh2, Vh2 = A @ Wqh2.T, A @ Wkh2.T, A @ Wvh2.T
print(Qh1.size(), Kh1.size(), Vh1.size())

# matrix multiplication, no need to transpose: no .T
QKTh1 = Qh1 @ Kh1.T
QKTh2 = Qh2 @ Kh2.T
print(QKTh1.size(), QKTh2.size())

QKTh1 = torch.dropout(torch.softmax(QKTh1 / math.sqrt(Kh1.size(-1)), -1), 0.1, False)
QKTh2 = torch.dropout(torch.softmax(QKTh2 / math.sqrt(Kh2.size(-1)), -1), 0.1, False)

# matrix multiplication, no need to transpose: no .T
QKVh1 = QKTh1 @ Vh1
QKVh2 = QKTh2 @ Vh2
print(QKVh1, QKVh2)

# model weights of layer, needs to transpose .T
output_h1 = QKVh1 @ Woh1.T
output_h2 = QKVh2 @ Woh2.T

output_MHA_TP = output_h1 + output_h2  # sum operation
# data synchronization through AllReduce communication in distributed system
# then all device share the sum of partitions
print('output_MHA_TP', output_MHA_TP)
# end [TP MHA]

with torch.no_grad():
    MHA_output, _ = attention_layer(A, A, A)
    print('MHA_output', MHA_output)
    print(torch.allclose(MHA_output, output_MHA_TP))

# connective block: central execution, omitted

# MLP block
rate = 4
MLP_layer = PositionwiseFeedForward(dims=hid, rate=rate)
A = torch.randn(seq, hid)
# begin [TP MLP]

# layers weights
activation = MLP_layer[1]
dropout = MLP_layer[2]
W1 = MLP_layer[0].weight
W2 = MLP_layer[-1].weight
print(W1.shape, W2.shape)

# weights partition
div_position = (rate * hid) // 2
W1p1 = W1[:div_position, :]
W1p2 = W1[div_position:, :]
W2p1 = W2[:, :div_position]
W2p2 = W2[:, div_position:]

# linear layer 1: need to transpose .T
O1p1 = A @ W1p1.T
O1p2 = A @ W1p2.T

# activation layer and dropout
I2p1 = dropout(activation(O1p1))
I2p2 = dropout(activation(O1p2))
# I2p1 = activation(O1p1)
# I2p2 = activation(O1p2)

# linear layer 2: need to transpose .T
O2p1 = I2p1 @ W2p1.T
O2p2 = I2p2 @ W2p2.T

# sum all partial outputs
output_MLP_TP = O2p1 + O2p2
print('output_MLP_TP', output_MLP_TP)

with torch.no_grad():
    MLP_output = MLP_layer(A)
    print('MLP_output', MLP_output)
    print(torch.allclose(MLP_output, output_MLP_TP))
    # **The results are not equal because of the dropout operation**
    # **When dropout is removed, the results are equal**

# base attention
# QKT = Q @ K.T
# element_wise_output = torch.softmax(torch.dropout(A, 0.1, False), -1)
#
# output_attention = element_wise_output @ Wo
# print(output_attention)
# print(output_attention.shape)
