import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

Past = Tuple[torch.Tensor, torch.Tensor]


class BaseAttention(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        print('QKT', x)
        # print('Divide', k.size(-1))

        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-1e4)
        # x = self.dropout(x.softmax(-1))
        x = x.softmax(-1)
        print('QKT', x)

        attention_output = torch.matmul(x, v)
        print('QKV', attention_output)
        # print(attention_output)
        return attention_output


class MultiHeadAttention(BaseAttention):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """
    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Split the tensors to multi-heads: , dims) -> , heads, dims//2)
        q_heads_shape = q.size()[:-1] + (self.heads, q.size(-1) // self.heads)
        k_heads_shape = k.size()[:-1] + (self.heads, k.size(-1) // self.heads)
        v_heads_shape = v.size()[:-1] + (self.heads, v.size(-1) // self.heads)
        print('Q_transformed', q_heads_shape)
        print('K_transformed', k_heads_shape)
        print('V_transformed', v_heads_shape)
        q = q.view(q_heads_shape)
        k = k.view(k_heads_shape)
        v = v.view(v_heads_shape)
        # then -3 is the head dimension
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        print('Transposed QKV')
        print('Q', q)
        print('K', k)
        print('V', v)

        if mask is not None:
            mask = mask.unsqueeze(-3)

        # Calculate multi-headed attentions and merge them into one.
        return (super().forward(q, k, v, mask)
                .transpose(-3, -2)
                .contiguous()  # for view (reshape) after transpose
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads)))


class AttentionLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., query_len, past_len + kv_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., query_len, dims)
    output 2 (*)    float           (..., past_len + kv_len, dims)
    ===========================================================================
    """
    def __init__(self, heads: int, dims: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims, bias=False)
        self.proj_k = nn.Linear(dims, dims, bias=False)
        self.proj_v = nn.Linear(dims, dims, bias=False)
        self.linear = nn.Linear(dims, dims, bias=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Past]:
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        print('Q', q)
        print('K', k)
        print('V', v)
        # Reuse attention keys and values by concatenating to the current ones.
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        x = self.attn(q, k, v, mask)
        x = self.linear(x)
        return x, (k, v)
