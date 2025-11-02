# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN


try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor

from contextlib import nullcontext
from profiler.profiler import prof

import numpy as np



# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if hasattr(config, "qkv_bias"):
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy  # (also tried x+self.dummy)


def len_list(x, n):
    return [i for i in x if len(i) <= n]


# draft model of eagle
class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",) as f:
                            #    device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1  # including the root node?
        self.depth = depth
        self.threshold = math.log(threshold)
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)

        self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()

        # if std is not None:
        #     noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
        #     inputs_embeds=inputs_embeds+noise

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        #position_ids=position_ids//4
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None
        
    def topk_1d(self, a: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if k < 1 or k > a.size:
            raise ValueError(f"k must be between 1 and {a.size}, got k={k}")
        # 部分划分取前 k 大的索引（无序）
        idx_part = np.argpartition(-a, k-1)[:k]      
        # 在这 k 个里做一次排序
        order = np.argsort(-a[idx_part])
        topk_idx = idx_part[order]
        topk_vals = a[topk_idx]
        return topk_idx, topk_vals

    def topk_2d_last(self, a: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        B, N = a.shape
        if k < 1 or k > N:
            raise ValueError(f"k must be between 1 and {N}, got k={k}")
        # 部分划分：每行前 k 大下标（无序）
        idx_part = np.argpartition(-a, k-1, axis=1)[:, :k]   # (B, k)
        # 行索引用于取值和二次排序
        rows = np.arange(B)[:, None]
        # 取出这 k 个的值
        vals_part = a[rows, idx_part]                       # (B, k)
        # 在每行这 k 个值里做降序排序
        order = np.argsort(-vals_part, axis=1)              # (B, k)
        topk_idx = idx_part[rows, order]                    # (B, k)
        topk_vals = vals_part[rows, order]                  # (B, k)
        return topk_idx, topk_vals

    def topk_np(self, a: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if a.ndim == 1:
            return self.topk_1d(a, k)
        elif a.ndim == 2:
            return self.topk_2d_last(a, k)
        else:
            raise ValueError("只支持 1D 或 2D 数组沿最后一维取 Top-K")

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor,
                     total_tokens=None, depth=None, top_k=None,
                     return_last=False,
                     log=False,
                     sort_score=False,
                     prof=None):
        """
        past_key_values **after draft-many** includes draft tokens
        """
        
        input_ids = input_ids.to(hidden_states.device)
        # [MODIFIED] custom tree scale
        if total_tokens is None:
            total_tokens = self.total_tokens
        if depth is None:
            depth = self.depth
        if top_k is None:
            top_k = self.top_k
        elif top_k != self.top_k:
            self.top_k = top_k
            self.init_tree()
            # print(f'top_k: {top_k}')

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # with Timer("draft many"):
        with prof.time_context(f"Stage 0: topk draft one", cpu=False) if prof is not None else nullcontext():
            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len = self.stable_kv[0][0].shape[2]
                # print(f'kv_len in topk_genrate: {kv_len}')
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                                past_key_values=self.stable_kv, use_cache=True)
            else:
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        # print(f"topk_cs_index.device={topk_cs_index.device}")
                
        with prof.time_context(f"Stage 0: topk draft main loop", cpu=False) if prof is not None else nullcontext():
            for i in range(depth):
                with prof.time_context(f"Stage 0: topk draft one loop", cpu=False) if prof is not None else nullcontext():
                    self.tree_mask = tree_mask
                    position_ids = len_posi + self.position_ids
                    # with Timer("draft one"):
                    with prof.time_context(f"Stage 0: topk draft one", cpu=False) if prof is not None else nullcontext():
                        out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                                        position_ids=position_ids, use_cache=True)

                    len_posi += 1

                    # with Timer("sort1"):
                    bias1 = top_k if i > 0 else 0
                    bias2 = max(0, i - 1)
                    bias = 1 + top_k ** 2 * bias2 + bias1
                    parents = (topk_cs_index + bias)
                    parents_list.append(parents)

                    last_headout = head(out_hidden[0])
                    last_p = self.logsoftmax(last_headout)

                    top = torch.topk(last_p, top_k, dim=-1)
                    topk_index, topk_p = top.indices, top.values

                    cu_scores = topk_p + scores[:, None]

                    topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
                    topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
                    scores = topk_cs_p

                    if log:
                        print(f'topK_genrate depth {i}:')
                        print(f'-- out_hidden: {out_hidden.shape}')
                        print(f'-- input_ids: {input_ids.shape}')
                        print(f'-- last_p: {last_p.shape}')
                        print(f'-- topk_p: {topk_p.shape}')
                        print(f'-- scores: {scores.shape}')
                        print(f'-- ss_token: {sum([i.numel() for i in ss_token])}')

                    out_ids = topk_cs_index // top_k
                    # print(f"out_ids.device={out_ids.device}")
                    input_hidden = out_hidden[:, out_ids]


                    # with Timer("2index"):
                    #     in_ids = topk_cs_index % top_k
                    #     input_ids = topk_index[out_ids, in_ids][None]
                    # with Timer("1index"):
                    input_ids = topk_index.view(-1)[topk_cs_index][None]
                    # print(input_ids.equal(input_ids0))

                    ss_token.append(topk_index)
                    scores_list.append(cu_scores)
                    # print(f"tree_mask.device={tree_mask.device}, self.tree_mask_init.device={self.tree_mask_init.device}")
                    # print(f"out_ids.device={out_ids.device}, input_ids.device={input_ids.device}")
                    tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

                # if self.threshold < 0 and cu_scores.max() < self.threshold:
                #     break  
            current_state = None
            if return_last:
                last_depth = depth
                current_state = (
                    last_depth, #total_tokens,
                    input_ids, input_hidden, past_key_values,
                    tree_mask, len_posi, top_k,
                    topk_cs_index, scores, ss_token, scores_list, parents_list,
                )
            
            scores_list = torch.cat(scores_list, dim=0).view(-1)
            ss_token_list = torch.cat(ss_token, dim=0).view(-1)
            parents_list = torch.cat(parents_list, dim=0).view(-1)
            
            # top_scores_index, _ = self.topk_np(scores_list, total_tokens)
            top_scores = torch.topk(scores_list, total_tokens, dim=-1, sorted=True)
            top_scores_index = top_scores.indices.cpu()
            
            # if return_last:
            #     current_state = current_state + (top_scores_index,)
            
            scores_list = scores_list.cpu()
            ss_token_list = ss_token_list.cpu()
            parents_list = parents_list.cpu()
            
        with prof.time_context(f"Stage 0: topk draft post process", cpu=True) if prof is not None else nullcontext():
            with prof.time_context(f"Stage 0: topk draft numpy", cpu=True) if prof is not None else nullcontext():
                scores_list = scores_list.numpy()
                ss_token_list = ss_token_list.numpy()
                parents_list = parents_list.numpy()
            
            top_scores_index = top_scores_index.numpy()
            
            if sort_score:
                # correct the top_scores_index
                top_scores_values = top_scores.values.cpu()
                top_scores_values = top_scores_values.numpy()
                sort_keys = np.column_stack((-top_scores_values, top_scores_index))
                lex_indices = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))
                top_scores_index = top_scores_index[lex_indices]
                
                draft_tokens = ss_token_list[top_scores_index]
                if return_last:
                    current_state = current_state + (top_scores_index,)
            
            # if sort_score:
            #     draft_tokens = ss_token_list[top_scores_index]
            #     if return_last:
            #         current_state = current_state + (top_scores_index,)

            # resort the top_scores_index
            top_orig_indices = np.argsort(top_scores_index)  # 保留原始顺序
            top_scores_index = top_scores_index[top_orig_indices]
            # top_scores_index = np.sort(top_scores_index)

            if not sort_score:
                draft_tokens = ss_token_list[top_scores_index]
                if return_last:
                    current_state = current_state + (top_scores_index,)
            else:
                top_orig_indices = np.pad(top_orig_indices+1, (1, 0), mode='constant', constant_values=0)
                # 构造反向索引
                inv_indices = np.zeros(top_orig_indices.size, dtype=np.int64)
                inv_indices[top_orig_indices] = np.arange(top_orig_indices.size, dtype=np.int64)

            # if return_last:
            #     current_state = current_state + (top_scores_index,)

            # draft_tokens = ss_token_list[top_scores_index]
            sample_token = sample_token.cpu().numpy()
            draft_tokens = np.concatenate((sample_token, draft_tokens))

            draft_parents = parents_list[top_scores_index // top_k].astype(np.int64)

            if log:
                print(f'draft_parents: {draft_parents}')
                print(f'ss_token_list: {ss_token_list.shape}')

            with prof.time_context(f"Stage 0: topk draft searchsorted", cpu=True) if prof is not None else nullcontext():
                mask_index = np.searchsorted(top_scores_index, draft_parents - 1, side='left')
            # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
            mask_index[draft_parents == 0] = -1
            mask_index = mask_index + 1
            mask_index_list = mask_index.tolist()
            # with Timer("mask"):
            tree_mask = np.eye(total_tokens + 1).astype(bool)
            tree_mask[:, 0] = True
            with prof.time_context(f"Stage 0: topk mask", cpu=True) if prof is not None else nullcontext():
                for i in range(total_tokens):
                    np.add(tree_mask[i+1], tree_mask[ mask_index_list[i] ], out=tree_mask[i+1])
                
            # with Timer("mask1"):
            #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
            #     tree_mask0[0][0] = True
            #     for i in range(total_tokens):
            #         #tree_mask0[i + 1][0]=True
            #         tree_mask0[i + 1][i + 1] = True
            #         p=mask_index_list[i]
            #         tree_mask0[i + 1][p] = True
            #         while p:
            #             p=mask_index_list[p-1]
            #             tree_mask0[i + 1][p] = True
            #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
            #
            # print(tree_mask0.equal(tree_mask))
                tree_position_ids = np.sum(tree_mask, axis=1) - 1

                # [update]
                if sort_score:
                    tree_mask = tree_mask[inv_indices]
                    tree_mask = tree_mask[:, inv_indices]

                tree_mask = tree_mask.astype(float)[None, None]
                draft_tokens = draft_tokens[None]

            del parents_list, scores_list, ss_token, ss_token_list, draft_parents

            # with Timer("retrieve"):

            max_depth = np.max(tree_position_ids) + 1
            noleaf_index = np.unique(mask_index).tolist()
            noleaf_num = len(noleaf_index) - 1
            leaf_num = total_tokens - noleaf_num

            retrieve_indices = np.full((leaf_num, max_depth), -1, dtype=np.int64)
            retrieve_indices = retrieve_indices.tolist()

            rid = 0
            position_ids_list = tree_position_ids.tolist()

            with prof.time_context(f"Stage 0: topk retrieve", cpu=True) if prof is not None else nullcontext():
                for i in range(total_tokens + 1):
                    if i not in noleaf_index:
                        cid = i
                        depth = position_ids_list[i]
                        for j in reversed(range(depth + 1)):
                            retrieve_indices[rid][j] = cid
                            cid = mask_index_list[cid - 1]
                        rid += 1

            with prof.time_context(f"Stage 0: topk sort", cpu=True) if prof is not None else nullcontext():
                if logits_processor is not None:
                    maxitem = total_tokens + 5

                    def custom_sort(lst):
                        # sort_keys=[len(list)]
                        sort_keys = []
                        for i in range(len(lst)):
                            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                        return sort_keys

                    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
                    
            with prof.time_context(f"Stage 0: topk tensor", cpu=True) if prof is not None else nullcontext():
                retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
                draft_tokens = torch.tensor(draft_tokens, dtype=torch.long)
                tree_mask = torch.tensor(tree_mask, dtype=torch.float)
                tree_position_ids = torch.tensor(tree_position_ids, dtype=torch.long)
            
                del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid

                # [update]
                if sort_score:
                    from pipeline_utils import map_retrieve_indices
                    retrieve_indices = map_retrieve_indices(retrieve_indices, torch.arange(draft_tokens.size(-1)), torch.from_numpy(top_orig_indices))
                    # retrieve_indices = map_retrieve_indices(retrieve_indices, top_orig_indices, top_scores_index)
                    tree_position_ids = tree_position_ids[inv_indices]
            
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, current_state
    
        # with prof.time_context(f"Stage 0: topk draft post process", cpu=False) if prof is not None else nullcontext():
        #     current_state = None
        #     if return_last:
        #         last_depth = i + 1
        #         current_state = (
        #             last_depth, #total_tokens,
        #             input_ids, input_hidden, past_key_values,
        #             tree_mask, len_posi, top_k,
        #             topk_cs_index, scores, ss_token, scores_list, parents_list,
        #         )
        #     # del parents_list,scores_list,ss_token
        #     # return draft_tokens, mask_index,tree_mask,tree_position_ids

        #     # with Timer("post"):

        #     scores_list = torch.cat(scores_list, dim=0).view(-1)
        #     ss_token_list = torch.cat(ss_token, dim=0).view(-1)

        #     # all_draft_size = scores_list.size(-1)
        #     # print(f'All draft: {all_draft_size}')

        #     top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        #     top_scores_index = top_scores.indices
        #     top_scores_index = torch.sort(top_scores_index).values

        #     if return_last:
        #         current_state = current_state + (top_scores_index,)

        #     draft_tokens = ss_token_list[top_scores_index]
        #     draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        #     draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        #     # print(f'draft_parents: {draft_parents}')

        #     if log:
        #         print(f'draft_parents: {draft_parents}')
        #         print(f'ss_token_list: {ss_token_list.shape}')

        #     mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        #     # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        #     mask_index[draft_parents == 0] = -1
        #     mask_index = mask_index + 1
        #     # print(f'mask_index:')
        #     # for i in range(len(mask_index)):
        #     #     print(f'{i+1} parent: {mask_index[i]}')
        #     mask_index_list = mask_index.tolist()
        #     # with Timer("mask"):
        #     tree_mask = torch.eye(total_tokens + 1).bool()
        #     tree_mask[:, 0] = True
        #     for i in range(total_tokens):
        #         tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        #     # with Timer("mask1"):
        #     #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     #     tree_mask0[0][0] = True
        #     #     for i in range(total_tokens):
        #     #         #tree_mask0[i + 1][0]=True
        #     #         tree_mask0[i + 1][i + 1] = True
        #     #         p=mask_index_list[i]
        #     #         tree_mask0[i + 1][p] = True
        #     #         while p:
        #     #             p=mask_index_list[p-1]
        #     #             tree_mask0[i + 1][p] = True
        #     #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #     #
        #     # print(tree_mask0.equal(tree_mask))
        #     tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        #     tree_mask = tree_mask.float()[None, None]
        #     draft_tokens = draft_tokens[None]



        #     del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        #     # with Timer("retrieve"):

        #     max_depth = torch.max(tree_position_ids) + 1
        #     noleaf_index = torch.unique(mask_index).tolist()
        #     noleaf_num = len(noleaf_index) - 1
        #     leaf_num = total_tokens - noleaf_num

        #     retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        #     retrieve_indices = retrieve_indices.tolist()

        #     rid = 0
        #     position_ids_list = tree_position_ids.tolist()

        #     for i in range(total_tokens + 1):
        #         if i not in noleaf_index:
        #             cid = i
        #             depth = position_ids_list[i]
        #             for j in reversed(range(depth + 1)):
        #                 retrieve_indices[rid][j] = cid
        #                 cid = mask_index_list[cid - 1]
        #             rid += 1

        #     if logits_processor is not None:
        #         maxitem = total_tokens + 5

        #         def custom_sort(lst):
        #             # sort_keys=[len(list)]
        #             sort_keys = []
        #             for i in range(len(lst)):
        #                 sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        #             return sort_keys

        #         retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        #     retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        #     del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        #     tree_position_ids = tree_position_ids.to(hidden_states.device)
        
        # return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, current_state
    
    @torch.no_grad()
    def expand_last_new(self, last_tree, last_state, head, logits_processor, device,
                    expand_depth=1, expand_size=20,
                    return_last=False, log=False,
                    prof=None):
        raise NotImplementedError("expand_last_new has some bugs, please use expand_last instead")
        """
        Expand the current tree with probs of all draft tokens
        """
        last_draft_tokens, last_retrieve_indices, last_tree_mask, last_tree_position_ids = last_tree
        last_depth, \
        input_ids, input_hidden, past_key_values, \
        tree_mask, len_posi, top_k, \
        topk_cs_index, scores, ss_token, scores_list, parents_list, last_top_scores_index = last_state
        
        total_tokens = last_top_scores_index.size + expand_size
        sample_token = input_ids[:, -1]
     
        with prof.time_context(f"Stage 0: expand last new main loop", cpu=False) if prof is not None else nullcontext():
            for i in range(last_depth, last_depth + expand_depth):
                with prof.time_context(f"Stage 0: expand last new draft one loop", cpu=False) if prof is not None else nullcontext():
                    self.tree_mask = tree_mask
                    position_ids = len_posi + self.position_ids
                    # with Timer("draft one"):
                    with prof.time_context(f"Stage 0: expand last new draft one", cpu=False) if prof is not None else nullcontext():
                        out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                                        position_ids=position_ids, use_cache=True)

                    len_posi += 1

                    # with Timer("sort1"):
                    bias1 = top_k if i > 0 else 0
                    bias2 = max(0, i - 1)
                    bias = 1 + top_k ** 2 * bias2 + bias1
                    parents = (topk_cs_index + bias)
                    parents_list.append(parents)

                    last_headout = head(out_hidden[0])
                    last_p = self.logsoftmax(last_headout)

                    top = torch.topk(last_p, top_k, dim=-1)
                    topk_index, topk_p = top.indices, top.values

                    cu_scores = topk_p + scores[:, None]

                    topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
                    topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
                    scores = topk_cs_p

                    # if log:
                    #     print(f'topK_genrate depth {i}:')
                    #     print(f'-- out_hidden: {out_hidden.shape}')
                    #     print(f'-- input_ids: {input_ids.shape}')
                    #     print(f'-- last_p: {last_p.shape}')
                    #     print(f'-- topk_p: {topk_p.shape}')
                    #     print(f'-- scores: {scores.shape}')
                    #     print(f'-- ss_token: {sum([i.numel() for i in ss_token])}')

                    out_ids = topk_cs_index // top_k
                    # print(f"out_ids.device={out_ids.device}")
                    input_hidden = out_hidden[:, out_ids]


                    # with Timer("2index"):
                    #     in_ids = topk_cs_index % top_k
                    #     input_ids = topk_index[out_ids, in_ids][None]
                    # with Timer("1index"):
                    input_ids = topk_index.view(-1)[topk_cs_index][None]
                    # print(input_ids.equal(input_ids0))

                    ss_token.append(topk_index)
                    scores_list.append(cu_scores)
                    # print(f"tree_mask.device={tree_mask.device}, self.tree_mask_init.device={self.tree_mask_init.device}")
                    # print(f"out_ids.device={out_ids.device}, input_ids.device={input_ids.device}")
                    tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
                
            if return_last:
                if expand_depth > 0:
                    last_depth = i + 1
                current_state = (
                    last_depth, #total_tokens,
                    input_ids, input_hidden, past_key_values,
                    tree_mask, len_posi, top_k,
                    topk_cs_index, scores, ss_token, scores_list, parents_list,
                )
            
            scores_list = torch.cat(scores_list, dim=0).view(-1)
            ss_token_list = torch.cat(ss_token, dim=0).view(-1)
            parents_list = torch.cat(parents_list, dim=0).view(-1)
            
            # top_scores_index, _ = self.topk_np(scores_list, total_tokens)
            top_scores = torch.topk(scores_list, total_tokens, dim=-1)
            top_scores_index = top_scores.indices.cpu().numpy()

            # print(f'top_scores_index: {top_scores_index}')
            # print(f'last_top_scores_index: {last_top_scores_index}')
            # new = set(top_scores_index.tolist()) - set(last_top_scores_index.tolist())
            # print(f'new: {new}, {len(new)}')
            # exit(0)
            
            if return_last:
                current_state = current_state + (top_scores_index,)
            
            scores_list = scores_list.cpu()
            ss_token_list = ss_token_list.cpu()
            parents_list = parents_list.cpu()
            
        with prof.time_context(f"Stage 0: expand last new post process", cpu=True) if prof is not None else nullcontext():
            scores_list = scores_list.numpy()
            ss_token_list = ss_token_list.numpy()
            parents_list = parents_list.numpy()
            
            top_scores_index = np.sort(top_scores_index)

            draft_tokens = ss_token_list[top_scores_index]
            sample_token = sample_token.cpu().numpy()
            draft_tokens = np.concatenate((sample_token, draft_tokens))
            

            draft_parents = parents_list[top_scores_index // top_k].astype(np.int64)
            # print(f'draft_parents: {draft_parents}')

            if log:
                print(f'draft_parents: {draft_parents}')
                print(f'ss_token_list: {ss_token_list.shape}')

            with prof.time_context(f"Stage 0: expand last new searchsorted", cpu=True) if prof is not None else nullcontext():
                mask_index = np.searchsorted(top_scores_index, draft_parents - 1, side='left')
            # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
            mask_index[draft_parents == 0] = -1
            mask_index = mask_index + 1
            # print(f'mask_index:')
            # for i in range(len(mask_index)):
            #     print(f'{i+1} parent: {mask_index[i]}')
            mask_index_list = mask_index.tolist()
            # with Timer("mask"):
            tree_mask = np.eye(total_tokens + 1).astype(bool)
            tree_mask[:, 0] = True
            with prof.time_context(f"Stage 0: expand last new mask", cpu=True) if prof is not None else nullcontext():
                for i in range(total_tokens):
                    np.add(tree_mask[i+1], tree_mask[ mask_index_list[i] ], out=tree_mask[i+1])
                    
                # with Timer("mask1"):
                #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
                #     tree_mask0[0][0] = True
                #     for i in range(total_tokens):
                #         #tree_mask0[i + 1][0]=True
                #         tree_mask0[i + 1][i + 1] = True
                #         p=mask_index_list[i]
                #         tree_mask0[i + 1][p] = True
                #         while p:
                #             p=mask_index_list[p-1]
                #             tree_mask0[i + 1][p] = True
                #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
                #
                # print(tree_mask0.equal(tree_mask))
                tree_position_ids = np.sum(tree_mask, axis=1) - 1

                tree_mask = tree_mask.astype(float)[None, None]
                draft_tokens = draft_tokens[None]



            del parents_list, scores_list, ss_token, ss_token_list, draft_parents

            # with Timer("retrieve"):

            max_depth = np.max(tree_position_ids) + 1
            noleaf_index = np.unique(mask_index).tolist()
            noleaf_num = len(noleaf_index) - 1
            leaf_num = total_tokens - noleaf_num

            retrieve_indices = np.full((leaf_num, max_depth), -1, dtype=np.int64)
            retrieve_indices = retrieve_indices.tolist()

            rid = 0
            position_ids_list = tree_position_ids.tolist()

            with prof.time_context(f"Stage 0: expand last new retrieve", cpu=True) if prof is not None else nullcontext():
                for i in range(total_tokens + 1):
                    if i not in noleaf_index:
                        cid = i
                        depth = position_ids_list[i]
                        for j in reversed(range(depth + 1)):
                            retrieve_indices[rid][j] = cid
                            cid = mask_index_list[cid - 1]
                        rid += 1
            
            with prof.time_context(f"Stage 0: expand last new sort", cpu=True) if prof is not None else nullcontext():
                if logits_processor is not None:
                    maxitem = total_tokens + 5

                    def custom_sort(lst):
                        # sort_keys=[len(list)]
                        sort_keys = []
                        for i in range(len(lst)):
                            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                        return sort_keys

                    retrieve_indices = sorted(retrieve_indices, key=custom_sort)

            with prof.time_context(f"Stage 0: expand last new tensor", cpu=True) if prof is not None else nullcontext():
                retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
                draft_tokens = torch.tensor(draft_tokens, dtype=torch.long)
                tree_mask = torch.tensor(tree_mask, dtype=torch.float)
                tree_position_ids = torch.tensor(tree_position_ids, dtype=torch.long)
                
                del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
                
                # assert retrieve_indices.device == torch.device(f"cuda:{1}")
                # assert draft_tokens.device == torch.device(f"cuda:{1}")
                # assert tree_mask.device == torch.device(f"cuda:{1}")
                # assert tree_position_ids.device == torch.device(f"cuda:{1}")
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, current_state  
        # with prof.time_context(f"Stage 0: expand last new post process", cpu=False) if prof is not None else nullcontext():
        #     current_state = None
        #     if return_last:
        #         last_depth = i + 1
        #         current_state = (
        #             last_depth,
        #             input_ids, input_hidden, past_key_values,
        #             tree_mask, len_posi, top_k,
        #             topk_cs_index, scores, ss_token, scores_list, parents_list,
        #         )
        #     # del parents_list,scores_list,ss_token
        #     # return draft_tokens, mask_index,tree_mask,tree_position_ids

        #     # with Timer("post"):

        #     scores_list = torch.cat(scores_list, dim=0).view(-1)
        #     ss_token_list = torch.cat(ss_token, dim=0).view(-1)

        #     # all_draft_size = scores_list.size(-1)
        #     # print(f'All draft: {all_draft_size}')

        #     top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        #     top_scores_index = top_scores.indices
        #     top_scores_index = torch.sort(top_scores_index).values

        #     if return_last:
        #         current_state = current_state + (top_scores_index,)

        #     draft_tokens = ss_token_list[top_scores_index]
        #     draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        #     draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        #     # print(f'draft_parents: {draft_parents}')

        #     if log:
        #         print(f'draft_parents: {draft_parents}')
        #         print(f'ss_token_list: {ss_token_list.shape}')

        #     mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        #     # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        #     mask_index[draft_parents == 0] = -1
        #     mask_index = mask_index + 1
        #     # print(f'mask_index:')
        #     # for i in range(len(mask_index)):
        #     #     print(f'{i+1} parent: {mask_index[i]}')
        #     mask_index_list = mask_index.tolist()
        #     # with Timer("mask"):
        #     tree_mask = torch.eye(total_tokens + 1).bool()
        #     tree_mask[:, 0] = True
        #     for i in range(total_tokens):
        #         tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        #     # with Timer("mask1"):
        #     #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     #     tree_mask0[0][0] = True
        #     #     for i in range(total_tokens):
        #     #         #tree_mask0[i + 1][0]=True
        #     #         tree_mask0[i + 1][i + 1] = True
        #     #         p=mask_index_list[i]
        #     #         tree_mask0[i + 1][p] = True
        #     #         while p:
        #     #             p=mask_index_list[p-1]
        #     #             tree_mask0[i + 1][p] = True
        #     #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #     #
        #     # print(tree_mask0.equal(tree_mask))
        #     tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        #     tree_mask = tree_mask.float()[None, None]
        #     draft_tokens = draft_tokens[None]



        #     del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        #     # with Timer("retrieve"):

        #     max_depth = torch.max(tree_position_ids) + 1
        #     noleaf_index = torch.unique(mask_index).tolist()
        #     noleaf_num = len(noleaf_index) - 1
        #     leaf_num = total_tokens - noleaf_num

        #     retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        #     retrieve_indices = retrieve_indices.tolist()

        #     rid = 0
        #     position_ids_list = tree_position_ids.tolist()

        #     for i in range(total_tokens + 1):
        #         if i not in noleaf_index:
        #             cid = i
        #             depth = position_ids_list[i]
        #             for j in reversed(range(depth + 1)):
        #                 retrieve_indices[rid][j] = cid
        #                 cid = mask_index_list[cid - 1]
        #             rid += 1

        #     if logits_processor is not None:
        #         maxitem = total_tokens + 5

        #         def custom_sort(lst):
        #             # sort_keys=[len(list)]
        #             sort_keys = []
        #             for i in range(len(lst)):
        #                 sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        #             return sort_keys

        #         retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        #     retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        #     del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        #     tree_position_ids = tree_position_ids.to(device)
        
        # return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, current_state
    
    @torch.no_grad()
    def expand_last(self, last_tree, last_state, head, logits_processor, device,
                    expand_depth=1, expand_size=20,
                    return_last=True, log=False,
                    prof=None):
        """
        Expand the current tree with probs of all draft tokens
        """
        last_draft_tokens, last_retrieve_indices, last_tree_mask, last_tree_position_ids = last_tree
        last_depth, \
        input_ids, input_hidden, past_key_values, \
        tree_mask, len_posi, top_k, \
        topk_cs_index, scores, ss_token, scores_list, parents_list, last_top_scores_index = last_state

        with prof.time_context(f"Stage 0: expand last main loop", cpu=False) if prof is not None else nullcontext():
            for i in range(last_depth, last_depth + expand_depth):
                self.tree_mask = tree_mask
                position_ids = len_posi + self.position_ids
                # with Timer("draft one"):
                with prof.time_context(f"Stage 0: expand last draft one", cpu=False) if prof is not None else nullcontext():
                    out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                                    position_ids=position_ids, use_cache=True)

                len_posi += 1

                # with Timer("sort1"):
                bias1 = top_k if i > 0 else 0
                bias2 = max(0, i - 1)
                bias = 1 + top_k ** 2 * bias2 + bias1
                parents = (topk_cs_index + bias)
                parents_list.append(parents)

                last_headout = head(out_hidden[0])
                last_p = self.logsoftmax(last_headout)

                top = torch.topk(last_p, top_k, dim=-1)
                topk_index, topk_p = top.indices, top.values

                cu_scores = topk_p + scores[:, None]

                topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
                topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
                scores = topk_cs_p

                out_ids = topk_cs_index // top_k
                # print(f"out_ids.device={out_ids.device}")
                input_hidden = out_hidden[:, out_ids]


                # with Timer("2index"):
                #     in_ids = topk_cs_index % top_k
                #     input_ids = topk_index[out_ids, in_ids][None]
                # with Timer("1index"):
                input_ids = topk_index.view(-1)[topk_cs_index][None]
                # print(input_ids.equal(input_ids0))

                ss_token.append(topk_index)
                scores_list.append(cu_scores)
                # print(f"tree_mask.device={tree_mask.device}, self.tree_mask_init.device={self.tree_mask_init.device}")
                # print(f"out_ids.device={out_ids.device}, input_ids.device={input_ids.device}")
                tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

        with prof.time_context(f"Stage 0: expand last post process", cpu=True) if prof is not None else nullcontext():
            if return_last:
                if expand_depth > 0:
                    last_depth = i + 1

                current_state = (
                    last_depth,
                    input_ids, input_hidden, past_key_values,
                    tree_mask, len_posi, top_k,
                    topk_cs_index, scores, ss_token, scores_list, parents_list
                )


            scores_list = torch.cat(scores_list, dim=0).view(-1).cpu().numpy()
            ss_token_list = torch.cat(ss_token, dim=0).view(-1).cpu()
            parents_list = torch.cat(parents_list, dim=0).view(-1).cpu().numpy()

            # all_draft_size = scores_list.size(-1)
            # mask last_selected
            # first_k_layers = 3  ###fixme: this may cause a disconnected tree
            # bias = top_k + (first_k_layers - 1) * top_k**2
            last_selected_mask = np.ones_like(scores_list, dtype=np.bool_)
            # last_selected_mask[:bias] = False
            last_selected_mask[last_top_scores_index] = False
            assert np.sum(last_selected_mask) > expand_size
            masked_scores_list = scores_list[last_selected_mask]
            
            # scores_list[last_top_scores_index] = -torch.inf
            # appended_top_scores = torch.topk(scores_list, expand_size, dim=-1)

            valid_indices = np.flatnonzero(last_selected_mask)  # selectable indices
            appended_top_scores = np.argsort(masked_scores_list)[-expand_size:]
            appended_top_scores_index = valid_indices[appended_top_scores]
            appended_top_scores_index = np.sort(appended_top_scores_index)
            # print(f'appended_top_scores_index: {appended_top_scores_index}')

            # scores_list = scores_list.cpu().numpy()
            # ss_token_list = ss_token_list.cpu().numpy()
            # parents_list = parents_list.cpu().numpy()

            last_size = last_draft_tokens.size(-1)

            merged_top_indices = np.concatenate((last_top_scores_index, appended_top_scores_index), axis=-1)
            if return_last:
                current_state = current_state + (merged_top_indices,)

            # scores_list = scores_list.cpu()
            # ss_token_list = ss_token_list.cpu()
            # parents_list = parents_list.cpu()

            merged_indices_origin = np.argsort(merged_top_indices)
            merged_sorted_top_indices = merged_top_indices[merged_indices_origin]
            
            # merged_sorted_top_indices, merged_indices_origin = np.sort(merged_top_indices)
            merged_indices_origin = np.pad(merged_indices_origin+1, (1, 0), mode='constant', constant_values=0)

            # 构造反向索引
            inv_indices = np.zeros(merged_indices_origin.size, dtype=np.int64)
            inv_indices[merged_indices_origin] = np.arange(merged_indices_origin.size, dtype=np.int64)

            draft_tokens = torch.cat((last_draft_tokens[0], ss_token_list[appended_top_scores_index]), dim=-1)
            
            draft_tokens_new = torch.cat((last_draft_tokens[:, 0], ss_token_list[merged_sorted_top_indices]), dim=-1)
            draft_tokens_new = draft_tokens_new[inv_indices]
            assert torch.equal(draft_tokens, draft_tokens_new)


            # print(f'parents_list: {parents_list}')
            draft_parents = parents_list[merged_sorted_top_indices // top_k].astype(np.int64)

            # test
            draft_parents_indices = draft_parents - 1
            draft_parents_indices[draft_parents_indices == -1] = 0
            
            parents_set = set(draft_parents_indices)
            selected_set = set(merged_sorted_top_indices)
            try:
                assert parents_set.issubset(selected_set)
            except:
                orig_parents = parents_list[merged_top_indices // top_k].astype(np.int64)
                orig_parents = orig_parents - 1
                
                # orig_parents[orig_parents == -1] = 0
                print(f'draft_parents_indices: {orig_parents}')
                print(f'merged_top_indices: {merged_top_indices}')
                # print(f'selected_set: {selected_set}')
                # print(f'parents_set: {parents_set}')
                print(f'diff: {parents_set - selected_set}')

                sorted_indices_global = np.argsort(-scores_list)
                # check_list = []
                for sorted_index in sorted_indices_global:
                    parent_idx = parents_list[sorted_index // top_k] - 1
                    if sorted_index in selected_set:
                        selected = 'Yes'
                    else:
                        selected = 'No'
                    print(f'{sorted_index:3d}: parent={parent_idx:3d}, selected={selected}')
                raise

            mask_index = np.searchsorted(merged_sorted_top_indices, draft_parents - 1, side='left')
            # mask_index = mask_index[torch.sort(merged_indices_origin).values]
            mask_index[draft_parents == 0] = -1
            mask_index = mask_index + 1

            # print(f'mask_index: {mask_index}')
            mask_index_list = mask_index.tolist()

            total_tokens = last_size + expand_size - 1
            tree_mask = np.eye(total_tokens + 1).astype(np.bool_)
            tree_mask[:, 0] = True
            for i in range(total_tokens):
                # tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
                # print(f'{i+1}, {mask_index_list[i]}')
                try:
                    np.add(tree_mask[i + 1], tree_mask[mask_index_list[i]], out=tree_mask[i + 1])
                except:
                    print(f'{i+1}, {mask_index_list[i]}')
                    raise

            # tree_mask[:last_size, :last_size] = last_tree_mask.cpu().numpy()
            # for i in range(last_size - 1, total_tokens):
            #     try:
            #         np.add(tree_mask[i + 1], tree_mask[mask_index_list[i]], out=tree_mask[i + 1])
            #     except:
            #         print(f'{i+1}, {mask_index_list[i]}')
            #         raise

            tree_position_ids = np.sum(tree_mask, axis=1) - 1
            # tree_position_ids = tree_position_ids[inv_indices]
            tree_mask = tree_mask[inv_indices]
            tree_mask = tree_mask[:, inv_indices]
            tree_mask = tree_mask.astype(np.float32)[None, None]

            tree_mask = torch.from_numpy(tree_mask)
            try:
                assert torch.allclose(tree_mask[0, 0, :last_size, :last_size], last_tree_mask[0, 0])
            except:
                print(f'last_size: {last_size}; expand_size: {expand_size}')
                print(f'tree_mask: {tree_mask[0, 0, :last_size, :last_size]}')
                print(f'last_tree_mask: {last_tree_mask[0, 0]}')
                diff = tree_mask[0, 0, :last_size, :last_size] - last_tree_mask[0, 0]
                print(f'diff: {diff.reshape(-1)}')
                print(f'diff.shape: {diff.shape}')
                print(f'positions: {torch.nonzero(diff, as_tuple=True)}')
                raise
            draft_tokens = draft_tokens[None]

            # [retrieve_indices]
            max_depth = np.max(tree_position_ids) + 1
            noleaf_index = np.unique(mask_index).tolist()
            noleaf_num = len(noleaf_index) - 1
            leaf_num = total_tokens - noleaf_num

            retrieve_indices = np.full((leaf_num, max_depth), -1, dtype=np.int64)
            retrieve_indices = retrieve_indices.tolist()

            rid = 0
            position_ids_list = tree_position_ids.tolist()
            tree_position_ids = torch.from_numpy(tree_position_ids[inv_indices])
            
            assert torch.allclose(tree_position_ids[:last_size], last_tree_position_ids)

            for i in range(total_tokens + 1):
                if i not in noleaf_index:
                    cid = i
                    depth = position_ids_list[i]
                    for j in reversed(range(depth + 1)):
                        retrieve_indices[rid][j] = cid
                        cid = mask_index_list[cid - 1]
                    rid += 1

            if logits_processor is not None:
                maxitem = total_tokens + 5

                def custom_sort(lst):
                    # sort_keys=[len(list)]
                    sort_keys = []
                    for i in range(len(lst)):
                        sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                    return sort_keys

                retrieve_indices = sorted(retrieve_indices, key=custom_sort)

            retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
            from pipeline_utils import map_retrieve_indices
            retrieve_indices = map_retrieve_indices(retrieve_indices, torch.arange(draft_tokens.size(-1)), torch.from_numpy(merged_indices_origin))
            del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid

            # tree_mask = torch.from(tree_mask, dtype=torch.float)
            # tree_position_ids = torch.tensor(tree_position_ids, dtype=torch.long)
            # tree_position_ids = tree_position_ids.to(device)
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, current_state

    @torch.no_grad()    
    def expand_pipedec(self, hidden_states, input_ids, head, logits_processor,
                       top_k=None,
                       log=False,
                       first_expand=False,
                       last_state=None, tree=None,
                       accept_tokens=None, left_indices=None):
        
        if top_k is None:
            top_k = self.top_k
        elif top_k != self.top_k:
            self.top_k = top_k
            self.init_tree()
        
        if first_expand:
            input_ids = input_ids.to(hidden_states.device)
            sample_token = input_ids[:, -1]
            scores_list = []
            parents_list = []
            input_hidden_list = []

            input_ids = input_ids[:, 1:]
            input_ids = input_ids.to(hidden_states.device)
            len_posi = input_ids.shape[1]
            self.reset()

            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len = self.stable_kv[0][0].shape[2]
                # print(f'kv_len in topk_genrate: {kv_len}')
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                                past_key_values=self.stable_kv, use_cache=True)
            else:
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)

            self.stable_kv = past_key_values
            # print(f'out_hidden: {out_hidden.shape}')
            last_hidden = out_hidden[:, -1]
            # print(f'last_hidden: {last_hidden.shape}')

            input_hidden = last_hidden[None].repeat(1, top_k, 1)
            # print(f'input_hidden: {input_hidden.shape}')
            # input_hidden_list.append(input_hidden)  # 下一次要用

            last_headout = head(last_hidden)
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p.view(-1), top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            scores = topk_p

            # 不包含树的根节点
            cu_scores_cum = topk_p
            # print(f'cu_scores: {cu_scores_cum.shape}')

            # draft_tokens = torch.cat((input_ids[:, :1], topk_index[None]), dim=-1)
            draft_tokens = torch.cat((sample_token[None], topk_index[None]), dim=-1)
            # first expand仅有两层，第二层的全部看得到第一层的根节点
            tree_mask = torch.eye(1+top_k, 1+top_k, dtype=torch.bool)
            tree_mask[:, 0] = True
            tree_mask = tree_mask[None, None].to(torch.float32)
            # 根节点为0，后面全为1
            tree_position_ids = torch.ones(1+top_k, dtype=torch.long)
            tree_position_ids[0] = 0

            # retrieve_indices
            retrieve_indices = torch.stack((torch.zeros(top_k, dtype=torch.long), torch.arange(1, top_k+1, dtype=torch.long)), dim=1)

            accept_hidden = None
            # current_state: 用于下次继续延申
            current_state = (
                input_hidden, len_posi, cu_scores_cum, accept_hidden
            )

            return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, current_state
        
        # [expand one layer]
        input_hidden, init_len_posi, cu_scores_cum, accept_hidden = last_state
        # input_hidden = torch.cat(input_hidden_list, dim=1)

        # 可能已经剪过枝了（如果剪过则accept_tokens非None以及left_indices非None
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = tree
        cur_input_len = input_ids.size(-1)
        # print(f'cur_tree_size: {tree_mask.shape}')

        # 当前树最后一层的节点在树中对应的indices
        max_position_id = torch.max(tree_position_ids)
        is_last_layer = tree_position_ids == max_position_id
        last_layer_indices = torch.nonzero(is_last_layer).squeeze(-1)
        # print(f'last_layer_indices: {last_layer_indices}')
        last_layer_size = is_last_layer.sum().item()

        tree_pos_ids_ea = tree_position_ids - 1

        # input_ids: 初始draft token tree对应的left_indices（不包含初始根节点）
        if accept_tokens is None:
            input_hidden_ea = input_hidden
            input_ids = draft_tokens[:, 1:]
            position_ids = tree_pos_ids_ea[1:]
            tree_mask_ea = tree_mask[:, :, 1:, 1:]
        else:
            # pruning the input_hidden and cu_scores_cum
            if accept_tokens.size(-1) > 1 and left_indices is not None:
                appended_accept_hidden = input_hidden[:, :1]
                if accept_hidden is None:
                    accept_hidden = appended_accept_hidden
                else:
                    accept_hidden = torch.cat((accept_hidden, appended_accept_hidden), dim=1)
            
            if left_indices is not None:
                # print(f'left_indices: {left_indices.shape}')

                left_indices_in_state = left_indices[1:] - 1
                # print(f'left_indices_in_state: {left_indices_in_state}')
                input_hidden = input_hidden[:, left_indices_in_state]
                # print(f'cu_scores_cum: {cu_scores_cum.shape}')
                cu_scores_cum = cu_scores_cum[left_indices_in_state]
            # print(f'input_hidden: {input_hidden.shape}, cu_scores_cum: {cu_scores_cum.shape}')
            # print(f'accept_tokens: {accept_tokens.size(-1)}')
            # if accept_hidden is not None:
                # print(f'accept_hidden: {accept_hidden.shape}')

            if accept_tokens.size(-1) == 1: # 直接把整棵树输进去，但是position要减1
                # print(f'accept_tokens: {accept_tokens.shape}')
                input_hidden_ea = input_hidden
                input_ids = draft_tokens
                position_ids = tree_pos_ids_ea
                tree_mask_ea = tree_mask
            else:
                input_hidden_ea = torch.cat((accept_hidden, input_hidden), dim=1)
                input_ids = torch.cat((accept_tokens[:, 1:], draft_tokens), dim=-1)
                

                # print(f'init_len_posi: {init_len_posi}, cur_input_len: {cur_input_len}')
                position_ids = torch.arange(init_len_posi, init_len_posi + accept_tokens.size(-1) - 1, dtype=torch.long)
                position_ids = torch.cat((position_ids, tree_pos_ids_ea), dim=0)
                accept_len = accept_tokens.size(-1) - 1
                tree_size = draft_tokens.size(-1)
                tree_mask_ea = torch.zeros(accept_len + tree_size, accept_len + tree_size, dtype=torch.float32)
                for i in range(accept_len):
                    tree_mask_ea[i:, i] = 1
                tree_mask_ea[accept_len:, accept_len:] = tree_mask
                tree_mask_ea = tree_mask_ea[None, None]
        
        self.tree_mask = tree_mask_ea



        # print(f'Stage 0: input_hidden_ea {input_hidden_ea.shape}, input_ids {input_ids.shape}, position_ids {position_ids.shape}, tree_mask {tree_mask_ea.shape}')
        # print(f'Stage 0: stable_kv: {self.stable_kv[0][0].shape}, position_ids: {position_ids}')
        assert input_hidden_ea.size(1) == position_ids.size(0) == input_ids.size(-1) == tree_mask_ea.size(-2)
        output_hidden, _ = self(input_hidden_ea, input_ids=input_ids,
                                            past_key_values=self.stable_kv,
                                            position_ids=position_ids, use_cache=True)
        # print(f'Stage 0: output_hidden {output_hidden.shape}')

        # 取得当前树最后一层的节点的next-token distribution
        last_layer_output_hidden = output_hidden[:, -last_layer_size:]
        last_headout = head(last_layer_output_hidden[0])
        last_layer_p = self.logsoftmax(last_headout)

        # print(f'Stage 0: last_layer_size {last_layer_size}, last_layer_p {last_layer_p.shape}')
        
        top = torch.topk(last_layer_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        
        last_layer_cu_scores = cu_scores_cum[-last_layer_size:]
        cu_scores = topk_p + last_layer_cu_scores[:, None]
        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values

        # out_ids = topk_cs_index // last_layer_size  # 如何映射回token_ids？
        out_ids = topk_cs_index // top_k

        topk_cs_index = topk_cs_index.cpu()
        # parents = topk_cs_index // 32000  # 相对于当前最后一层的tokens
        parents = topk_cs_index // top_k
        # print(f'parents: {parents}')

        # input_hidden_appended = last_layer_output_hidden[:, parents]
        input_hidden_appended = last_layer_output_hidden[:, out_ids]
        # print(f'input_hidden_appended: {input_hidden_appended.shape}')

        scores = topk_cs_p
        # print(f'scores: {scores.shape}')
        # 拿到这些parent在当前draft token tree中的序号
        # print(f'last_layer_indices: {last_layer_indices.shape}')
        parent_indices = last_layer_indices[parents]
        idx_ri_path = []  # parent_idx -> ri_path
        for parent_idx in last_layer_indices:
            ri_path = torch.nonzero(retrieve_indices[:, -1] == parent_idx).item()
            idx_ri_path.append(ri_path)

        # print(f'idx_ri_path: {idx_ri_path}')

        # 更新input_hidden和cu_scores_cum
        input_hidden = torch.cat((input_hidden, input_hidden_appended), dim=1)
        cu_scores_cum = torch.cat((cu_scores_cum, topk_cs_p), dim=-1)
        # print(f'appended input_hidden: {input_hidden.shape}, cu_scores_cum: {cu_scores_cum.shape}')

        last_tree_size = draft_tokens.size(-1)

        # 更新draft_tokens
        # print(f'out tokens: {self.tokenizer.decode(out_ids)}')
        # draft_tokens = torch.cat((draft_tokens, out_ids[None]), dim=-1)
        draft_tokens = torch.cat((draft_tokens, topk_index.view(-1)[topk_cs_index][None]), dim=-1)

        # 选出retrieve_indices中到底的路径并基于选中的节点进行延申
        # path_depths = retrieve_indices.sum(dim=1)
        # expand_choice = path_depths == max(path_depths)

        expanded_paths = torch.zeros(retrieve_indices.size(0), dtype=torch.bool)
        retrieve_indices = F.pad(retrieve_indices, (0, 1), value=-1)
        # kept_paths = retrieve_indices[~expand_choice]
        expand_paths = []
        for i in range(top_k):
            parent_ri_path = idx_ri_path[parents[i]]
            expanded_paths[parent_ri_path] = True
            new_path = retrieve_indices[parent_ri_path].clone()
            new_path[-1] = i + last_tree_size
            expand_paths.append(new_path)
        kept_paths = retrieve_indices[~expanded_paths]
        expand_paths = torch.stack(expand_paths, dim=0)

        # print(f'kept_paths: {kept_paths.shape}, expand_paths: {expand_paths.shape}')
        retrieve_indices = torch.cat((kept_paths, expand_paths), dim=0)

        # 更新tree_mask
        tree_mask_new = torch.eye(last_tree_size + top_k, last_tree_size + top_k, dtype=torch.float32)
        tree_mask_new[:last_tree_size, :last_tree_size] = tree_mask.clone()
        tree_mask_new[:, 0] = True
        for i in range(top_k):
            parent_index = parent_indices[i]
            tree_mask_new[last_tree_size+i] += tree_mask_new[parent_index]
        tree_mask_new = tree_mask_new[None, None]

        # 更新tree_position_ids
        max_pos_id = torch.max(tree_position_ids)
        appended_pos_ids = torch.full((top_k,), max_pos_id + 1, dtype=torch.long)
        tree_position_ids = torch.cat((tree_position_ids, appended_pos_ids), dim=0)

        # 更新input_hidden和cu_scores_cum
        last_state = (
            input_hidden,
            init_len_posi,
            cu_scores_cum,
            accept_hidden
        )

        return draft_tokens, retrieve_indices, tree_mask_new, tree_position_ids, last_state



    @torch.no_grad()
    def acc(self, data, head, max_length=5):
        hidden_states = data["hidden_states"]
        input_ids = data["input_ids"]
        # attention_mask=data["attention_mask"]
        loss_mask = data["loss_mask"]
        sample_mask = data["sample_mask"]
        target = data["target"]
        total = [0 for _ in range(max_length)]
        correct = [0 for _ in range(max_length)]
        bs, sl = hidden_states.shape[0], hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout = head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i, j] == 0:
                    continue
                single_hidden_states = hidden_states[i, :j]
                single_input_ids = input_ids[i, :j]

                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                    tmp_sample_mask = sample_mask[i, single_hidden_states.shape[1] - 1]
                    if not (target_in_token == tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token == target_out_token:
                        correct[k] += 1
                    else:
                        for kk in range(k, max_length):
                            total[kk] += 1
                        break

                    single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                    single_input_ids = torch.cat(
                        (single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

        acc = [correct[i] / total[i] for i in range(len(correct))]
        return acc


class Vhead(nn.Module):

    def __init__(self, ins=6566, outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins, outs, bias=False)

    def forward(self, x):
        return self.fc(x)


import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)
