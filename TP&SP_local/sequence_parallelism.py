# A simple example to show SEQUENCE PARALLELISM locally
# using the transformer layer of GPT-2
from layers import *
import math
import torch
import torch.nn.functional as F
import time

# when the size of input and hidden size are small, result is promised to be same as the local inference
seq = 4
hid = 4
h = heads = 2
N = 2  # division number
rate = 4  # the hidden dimension number in the feed-forward: rate * hid
# seed = time.time()
seed = 100
torch.manual_seed(int(seed))

# transformer layer
transformer_layer = TransformerLayer(heads=h, dims=hid, rate=rate)
transformer_layer.eval()
MHA_layer = transformer_layer.attn
A = torch.randn(seq, hid)  # input: embedded token
A = transformer_layer.ln_attn(A)
# original attention inference process
q, k, v = MHA_layer.proj_q(A), MHA_layer.proj_k(A), MHA_layer.proj_v(A)

multi_head_QKV_shape = q.size()[:-1] + (h, q.size(-1) // h)  # (seq, head, hid/head)
q_heads_shape = q.size()[:-1] + (h, q.size(-1) // h)
k_heads_shape = k.size()[:-1] + (h, k.size(-1) // h)
v_heads_shape = v.size()[:-1] + (h, v.size(-1) // h)
q = q.view(q_heads_shape)
k = k.view(k_heads_shape)
v = v.view(v_heads_shape)
# then -3 is the head dimension
q = q.transpose(-3, -2)
k = k.transpose(-3, -2)
v = v.transpose(-3, -2)
# print('Q', q)
# print('K', k)
# print('V', v)
x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
# print('QKT', x)
x = torch.dropout(x.softmax(-1), 0.1, False)
# x = x.softmax(-1)
# print('QKT', x)
o = torch.matmul(x, v)
# print('QKV', o)
o = o.transpose(-3, -2).contiguous().view(q.size()[:-3] + (q.size(-2), v.size(-1) * h))
o = MHA_layer.linear(o)
o1, _ = MHA_layer(A, A, A)
assert torch.allclose(o, o1)

# input division on the sequence dimension
divided_seq = seq // N
A1 = A[:divided_seq, :]
A2 = A[divided_seq:, :]

A1 = transformer_layer.ln_attn(A1)
A2 = transformer_layer.ln_attn(A2)

Wq, Wk, Wv, Wo = MHA_layer.proj_q.weight, MHA_layer.proj_k.weight, MHA_layer.proj_v.weight, MHA_layer.linear.weight

# linear layer: need transpose .T
Q, K, V = MHA_layer.proj_q(A), MHA_layer.proj_k(A), MHA_layer.proj_v(A)

multi_head_QKV_shape = Q.size()[:-1] + (h, Q.size(-1) // h)  # (seq, head, hid/head)
print('multi_head_QKV_shape', multi_head_QKV_shape)
Q = Q.view(multi_head_QKV_shape).transpose(-3, -2)  # (head, seq, hid/head)
K = K.view(multi_head_QKV_shape).transpose(-3, -2)
V = V.view(multi_head_QKV_shape).transpose(-3, -2)
# print(Q.shape, K.shape, V.shape)

# linear layer: need transpose .T
# get partial embeddings
Q1, Q2 = A1 @ Wq.T, A2 @ Wq.T
K1, K2 = A1 @ Wk.T, A2 @ Wk.T
V1, V2 = A1 @ Wv.T, A2 @ Wv.T


# reshape for multi-head attention
multi_head_QKVi_shape = Q1.size()[:-1] + (h, Q1.size(-1) // h)  # (seq/N, head, hid/head)
print('multi_head_QKVi_shape', multi_head_QKVi_shape)
# shift the head dimension to the front of seq and hid
# device 1
Q1 = Q1.view(multi_head_QKVi_shape).transpose(-3, -2)  # (head, seq/N, hid/head)
K1 = K1.view(multi_head_QKVi_shape).transpose(-3, -2)
V1 = V1.view(multi_head_QKVi_shape).transpose(-3, -2)
# device 2
Q2 = Q2.view(multi_head_QKVi_shape).transpose(-3, -2)
K2 = K2.view(multi_head_QKVi_shape).transpose(-3, -2)
V2 = V2.view(multi_head_QKVi_shape).transpose(-3, -2)
# print(Q1.shape, Q2.shape, K1.shape, K2.shape, V1.shape, V2.shape)
# QKV to test correctness
Q_SP = torch.concat((Q1, Q2), dim=1)
K_SP = torch.concat((K1, K2), dim=1)
V_SP = torch.concat((V1, V2), dim=1)
# assert torch.allclose(Q_SP, Q)
# assert torch.allclose(K_SP, K)
# assert torch.allclose(V_SP, V)


# 1. Sequence Parallelism
QKT = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.size(-1))
# matrix multiplication, no need to transpose: no .T
# device 1: while K1, K2 is on different devices, they are shared across device by communication
Q1K1 = torch.matmul(Q1, K1.transpose(-1, -2)) / math.sqrt(K1.size(-1))
Q1K2 = torch.matmul(Q1, K2.transpose(-1, -2)) / math.sqrt(K2.size(-1))
Q1KT = torch.concat((Q1K1, Q1K2), dim=-1)
# device 2
Q2K1 = torch.matmul(Q2, K1.transpose(-1, -2)) / math.sqrt(K1.size(-1))
Q2K2 = torch.matmul(Q2, K2.transpose(-1, -2)) / math.sqrt(K2.size(-1))
Q2KT = torch.concat((Q2K1, Q2K2), dim=-1)
print(Q1K1.shape, Q1K2.shape, Q2K1.shape, Q2K2.shape)

# Q1KT = torch.matmul(Q1, K.transpose(-1, -2))
print(Q1KT.shape, Q2KT.shape)

QKT_SP = torch.concat((Q1KT, Q2KT), dim=-2)
# assert torch.allclose(QKT, QKT_SP)

Q1KT = torch.dropout(torch.softmax(Q1KT, -1), 0.0, False)
Q2KT = torch.dropout(torch.softmax(Q2KT, -1), 0.0, False)

QKT = torch.dropout(torch.softmax(QKT, -1), 0.0, False)
QKT_SP = torch.concat((Q1KT, Q2KT), dim=-2)
# assert torch.allclose(QKT, QKT_SP)


QKTV = torch.matmul(QKT, V)
QKTV = QKTV.transpose(-3, -2).contiguous().view(Q.size()[:-3] + (Q.size(-2), V.size(-1) * h))


# split QiK in column, since Vi is partial
Q1K_ = [Q1KT[..., :seq//N], Q1KT[..., seq//N:]]
Q2K_ = [Q2KT[..., :seq//N], Q2KT[..., seq//N:]]
Vis = [V1, V2]

# matrix multiplication, no need to transpose: no .T
# device 1: while V1, V2 is on different devices, they are shared across devices by communication
O1 = sum([torch.matmul(Q1K_[i], Vis[i]) for i in range(N)])
O1 = O1.transpose(-3, -2)
O1 = O1.contiguous().view(Q1.size()[:-3] + (Q1.size(-2), V1.size(-1) * h))
# device 2
O2 = sum([torch.matmul(Q2K_[i], Vis[i]) for i in range(N)])
O2 = O2.transpose(-3, -2)
O2 = O2.contiguous().view(Q2.size()[:-3] + (Q2.size(-2), V2.size(-1) * h))
print(O1.shape, O2.shape)
QKTV_SP = torch.concat((O1, O2), dim=0)
# assert torch.allclose(QKTV, QKTV_SP)

output = MHA_layer.linear(QKTV)


# linear layer: need to transpose .T
O1 = torch.matmul(O1, Wo.T)
O2 = torch.matmul(O2, Wo.T)

output_MHA_SP = torch.concat((O1, O2), dim=0)  # concat on the seq dimension
# assert torch.allclose(output_MHA_SP, output)
# O = O.view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads)))
print('output_MHA_SP', output_MHA_SP)

with torch.no_grad():
    A = transformer_layer.ln_attn(A)
    MHA_output, _ = MHA_layer(A, A, A)
    print('MHA_output', MHA_output)
    print(torch.allclose(MHA_output, output_MHA_SP))

# connective block
# I1 = torch.dropout(O1, 0.1, False)  # dropout is moved to inside of MHA block and MLP block
# I2 = torch.dropout(O2, 0.1, False)
I1 = transformer_layer.ln_ff(O1)
I2 = transformer_layer.ln_ff(O2)

MLP_layer = transformer_layer.ff  # no need to partition
O1_MLP = MLP_layer(I1)
O2_MLP = MLP_layer(I2)

output_MLP_SP = torch.concat((O1_MLP, O2_MLP), dim=0)
print('output_MLP_SP', output_MLP_SP)
with torch.no_grad():
    I = MHA_output
    I = transformer_layer.ln_ff(I)
    MLP_output = MLP_layer(I)
    print('MLP_output', MLP_output)
    print(torch.allclose(MLP_output, output_MLP_SP))








