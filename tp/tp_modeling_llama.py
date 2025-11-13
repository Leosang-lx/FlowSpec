# import torch
# import torch.nn as nn
# import torch.distributed as dist
# from transformers import LlamaPreTrainedModel, LlamaConfig
# from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
# from transformers.models.llama.modeling_llama import (
#     LlamaRMSNorm,
#     LlamaDecoderLayer,
#     _make_causal_mask,
#     _expand_mask,
# )
# from safetensors.torch import load_file
# import os
# import math
# from tp.tp_layers import ColumnParallelLinear, RowParallelLinear

""" PyTorch LLaMA model."""
from typing import Optional, Tuple, Union
from contextlib import nullcontext
import torch
import torch.utils.checkpoint
from torch import nn

# [MODIFIED] Import from transformer library
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import (
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig
from eagle.modeling_llama_kv import LlamaPreTrainedModel, LlamaRMSNorm, LlamaAttention, LlamaMLP, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding # LlamaDecoderLayer
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from eagle.modeling_llama_kv import _expand_mask, _make_causal_mask, LlamaDecoderLayer
from tp.tp_layers import ColumnParallelLinear, RowParallelLinear
import torch.distributed as dist
import math
import torch.nn.functional as F
from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class TPLlamaAttention(nn.Module):
    """
    LlamaAttention is a multi-headed attention module based on the 'Attention Is All You Need' paper.

    Args:
        config (LlamaConfig): Configuration for the attention module.

    Attributes:
        config (LlamaConfig): Configuration for the attention module.
        hidden_size (int): The size of the hidden layer.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        num_key_value_heads (int): The number of key-value attention heads.
        num_key_value_groups (int): The number of key-value groups.
        pretraining_tp (int): The pretraining time periods.
        max_position_embeddings (int): The maximum position embeddings.

    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads // 4
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads * 4) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings,base=self.config.rope_theta
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

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

        if self.pretraining_tp > 1:
            key_value_slicing = (
                                        self.num_key_value_heads * self.head_dim
                                ) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # [MODIFIED] Using KVCache mechanism for preallocated GPU memory optimization
        # past_key_value is utilized to leverage previously computed key and value states.
        # If past_key_value is available, reuse the states for k, v, and self_attention.
        if past_key_value is not None:
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)
        # Reset past_key_value to avoid return past_key_value.
        past_key_value = None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # print(f"key_states.shape: {key_states.shape}")
        # print(f"value_states.shape: {value_states.shape}")
        # print(f"query_states.shape: {query_states.shape}")

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        # print(f"attn_weights.shape: {attn_weights.shape}")

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
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // 4)
        
        # print(f"attn_output.shape: {attn_output.shape}")

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class TPLlamaMLP(nn.Module):
    """
    LlamaMLP is a multi-layer perceptron module used in the Llama model.

    Args:
        config: The configuration for the MLP.

    Attributes:
        pretraining_tp (int): The pretraining time periods.
        hidden_size (int): The size of the hidden layer.
        intermediate_size (int): The size of the intermediate layer.
        gate_proj (nn.Linear): The linear projection for gating.
        up_proj (nn.Linear): The linear projection for the up projection.
        down_proj (nn.Linear): The linear projection for the down projection.
        act_fn: The activation function.

    """

    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // 4
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key and value tensors n times along the specified dimension.

    Args:
        hidden_states (torch.Tensor): Input tensor with shape (batch, num_key_value_heads, seqlen, head_dim).
        n_rep (int): Number of times to repeat.

    Returns:
        torch.Tensor: Repeated tensor with shape (batch, num_key_value_heads * n_rep, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class TPLlamaDecoderLayer(nn.Module):
    """
    LlamaDecoderLayer represents a single layer of the Llama decoder.

    Args:
        config (LlamaConfig): Configuration for the decoder layer.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        self_attn (LlamaAttention): Multi-headed self-attention module.
        mlp (LlamaMLP): Multi-layer perceptron module.
        input_layernorm (LlamaRMSNorm): Layer normalization for input.
        post_attention_layernorm (LlamaRMSNorm): Layer normalization after self-attention.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TPLlamaAttention(config=config)
        self.mlp = TPLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            tp_group: Optional[dist.ProcessGroup] = None,
            prof=None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Forward pass for the LlamaDecoderLayer.

        Args:
            hidden_states (torch.FloatTensor): Input tensor of shape `(batch, seq_len, embed_dim)`.
            attention_mask (torch.FloatTensor, optional): Attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (torch.LongTensor, optional): Positional IDs tensor.
            past_key_value (Tuple[torch.FloatTensor], optional): Cached past key and value projection states.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
            use_cache (bool, optional): If set to `True`, `past_key_values` key-value states are returned and can be
                used to speed up decoding.

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: Tuple containing:
                - hidden_states (torch.FloatTensor): Output tensor.
                - self_attn_weights (Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]): Self-attention weights if
                  `output_attentions` is `True`.
                - present_key_value (Optional[Tuple[torch.FloatTensor]]): Cached key and value projection states if
                  `use_cache` is `True`.
        """

        residual = hidden_states

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
        with prof.profile_context(f"Rank {dist.get_rank()}: all_reduce", device="cpu") if prof else nullcontext():
            dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=tp_group)
        hidden_states = residual + hidden_states
        

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        with prof.profile_context(f"Rank {dist.get_rank()}: all_reduce", device="cpu") if prof else nullcontext():
            dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=tp_group)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class TPLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, tp_rank: int = 0, tp_size: int = 1):
        super().__init__(config)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        if tp_rank == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        else:
            assert config.vocab_size % tp_size == 0
            self.vocab_start = config.vocab_size//tp_size * (tp_rank - 1)
            self.vocab_end = config.vocab_size//tp_size * tp_rank
            self.embed_tokens = nn.Embedding(self.vocab_end - self.vocab_start, config.hidden_size, config.pad_token_id)
            # print(f'rank: {tp_rank} start init layers, config.num_hidden_layers: {config.num_hidden_layers}')
            self.layers = nn.ModuleList([
                self._wrap_tp(TPLlamaDecoderLayer(config))
                for _ in range(config.num_hidden_layers)
            ])
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.tree_mask = None  # Optional tree attention mask

    def _wrap_tp(self, layer: nn.Module) -> nn.Module:
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                parent = layer
                keys = name.split('.')
                for k in keys[:-1]:
                    parent = getattr(parent, k)
                last_key = keys[-1]
                linear = getattr(parent, last_key)
                in_f, out_f = linear.in_features, linear.out_features
                bias = linear.bias is not None
                if last_key in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']:
                    new_linear = ColumnParallelLinear(in_f, out_f, bias=bias, tp_rank=self.tp_rank, tp_size=self.tp_size)
                elif last_key in ['o_proj', 'down_proj']:
                    new_linear = RowParallelLinear(in_f, out_f, bias=bias, tp_rank=self.tp_rank, tp_size=self.tp_size)
                else:
                    continue
                setattr(parent, last_key, new_linear)
        return layer

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        if self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_tgt_len = tree_mask.size(-2)
            tree_src_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_tgt_len:, -tree_src_len:][tree_mask == 0] = combined_attention_mask.min()

        return combined_attention_mask
    
    @torch.no_grad()
    def forward_layers(
        self,
        hidden_states,
        past_key_values,
        position_ids,
        attention_mask,
        output_hidden_states,
        output_attentions,
        use_cache,
        tp_group=None,
        prof=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            # print(f"rank: {self.tp_rank} start forward idx: {idx}")
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                tp_group=tp_group,
                prof=prof,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return hidden_states, next_decoder_cache, all_hidden_states, all_self_attns
    
    def get_ring_index(self, world_size: int, index: int, offset: int):  # ring index starts from 0
        ring_index = index + offset
        if ring_index < 0:
            ring_index += world_size
        elif ring_index >= world_size:
            ring_index %= world_size
        return ring_index
    
    @torch.no_grad()
    def ring_reduce_scatter_comp_overlap(self, xi: torch.Tensor, decoder_layer: LlamaDecoderLayer, tp_group=None, case=0):
        """
        overlap the ring reduce-scatter comm operation with the **previous** matrix-vector multiplication
        :return: split reduced results
        """
        rank = dist.get_rank(group=tp_group)
        world_size = dist.get_world_size(group=tp_group)

        if case == 0:
            o_proj = decoder_layer.self_attn.o_proj
        else:
            down_proj = decoder_layer.mlp.down_proj

        # W_in, W_out = Wi.shape
        # the whole weight matrix is split along row dimension(dim=0) across devices
        # assert xi.size(-1) == W_in and W_out % self.n_device == 0

        # further split partial weight along column dimension(dim=1)
        # TODO: split the weight matrix
        W_split_size = W_out // self.n_device
        Wi_split = Wi.split(W_split_size, dim=-1)

        comm_from_index = self.get_ring_index(world_size, rank, -1)
        from_tensor = torch.zeros(*xi.shape[:-1], W_split_size)
        comm_to_index = self.get_ring_index(world_size, rank, 1)

        for i in range(self.n_device - 1, -1, -1):  # reverse order from n-1 to 0

            comp_index = self.get_ring_index(self.rank, i)  # [rank-1, rank-2, ..., rank]
            # start receiving
            if i != self.n_device - 1:
                send_task = dist.isend(yi, dst=comm_to_index)
                recv_task = dist.irecv(from_tensor, src=comm_from_index)
            else:
                recv_task = None

            # partial computation
            yi = torch.matmul(xi, Wi_split[comp_index])

            # recv result from others
            if recv_task:
                # comment the send_task.wait() for distributed=True
                # if not distributed:
                send_task.wait(timeout=timeout_max)
                recv_task.wait(timeout=timeout_max)
                yi.add_(from_tensor)  # reduce
        # if bi is not None:
        #     yi.add_(bi)
        return yi

    @torch.no_grad()
    def ring_all_gather_comp_overlap(self, xi: torch.Tensor, decoder_layer: LlamaDecoderLayer, tp_group=None, case=0):
        """
        overlap the ring all-gather comm operation with the **following** matrix-vector multiplication
        :return: gathered results after the matrix-vector multiplication
        """
        rank = dist.get_rank(group=tp_group)
        world_size = dist.get_world_size(group=tp_group)

        if case == 0:  # attn
            q_proj, k_proj, v_proj = decoder_layer.self_attn.q_proj, decoder_layer.self_attn.k_proj, decoder_layer.self_attn.v_proj
        else:  # ffn
            gate_proj, up_proj, act_fn = decoder_layer.mlp.gate_proj, decoder_layer.mlp.up_proj, decoder_layer.mlp.act_fn

            
        # W_in, W_out = Wi.shape
        # assert xi.size(-1) == W_in and W_out % self.n_device == 0

        comm_from_index = self.get_ring_index(rank, -1)
        from_tensor = torch.zeros_like(xi)  # from_tensor for saving the received tensor
        to_tensor = xi  # to_tensor for comp and comm at each turn
        comm_to_index = self.get_ring_index(rank, 1)

        split_results = [None] * self.n_device

        for i in range(self.n_device - 1, -1, -1):  # reverse order from n-1 to 0
            comp_index = self.get_ring_index(world_size, rank, i)  # shift order according to rank and i
            if i > 0:
                # start sending
                send_task = dist.isend(to_tensor, dst=comm_to_index, group=tp_group)
                recv_task = dist.irecv(from_tensor, src=comm_from_index)

            # partial computation
            if case == 0:
                res = (q_proj(to_tensor), k_proj(to_tensor), v_proj(to_tensor))
                res = torch.cat(res, dim=-1)
            else:
                res = act_fn(gate_proj(to_tensor)) * up_proj(to_tensor)
            split_results[comp_index] = res
            # split_results[comp_index] = to_tensor @ Wi

            if i > 0:
                send_task.wait(timeout=timeout_max)
                recv_task.wait(timeout=timeout_max)
                # exchange reference after send_task and recv_task are both finished
                from_tensor, to_tensor = to_tensor, from_tensor
            else:
                assert None not in split_results
                return torch.cat(split_results, dim=-2)

    @torch.no_grad()
    def mha_with_rope(
        self,
        q,
        k,
        v,
        attention_mask,
        position_ids,
        past_key_value,
        bsz,
        q_len,
        output_attentions=False
    ):
        query_states = q.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = k.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = v.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)
        past_key_value = None

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // 4)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
        

    @torch.no_grad()
    def forward_layers_galaxy(
        self,
        hidden_states,
        past_key_values,
        position_ids,
        attention_mask,
        output_hidden_states,
        output_attentions,
        use_cache,
        tp_group=None,
        prof=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        rank = self.get_rank()
        world_size = self.get_world_size()
        seq_len = hidden_states.size(-2)
        assert seq_len % world_size == 0

        seq_split_len = seq_len // world_size
        seq_l, seq_r = (seq_split_len * rank, seq_split_len * (rank + 1))

        bsz, q_len, hidden_size = hidden_states.size()

        for idx, decoder_layer in enumerate(self.layers):
            merged_qkv_proj = ...

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if idx == 0:
                residual = hidden_states[..., seq_l:seq_r, :]
                hidden_states = decoder_layer.input_layernorm(hidden_states)

                qkv_slice = F.linear(hidden_states, merged_qkv_proj)
            else:
                qkv_slice = self.ring_all_gather_comp_overlap(hidden_states, decoder_layer, tp_group, 0)

            # get qkv
            q, k, v = tuple(torch.split(qkv_slice, hidden_size, dim=-1))

            # mha
            attn_output, attn_weights, past_key_value = self.mha_with_rope(
                q, k, v,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            if use_cache:
                next_decoder_cache += (past_key_value,)

            if output_attentions:
                all_self_attns += (attn_weights,)

            # Wo projection
            hidden_states = self.ring_reduce_scatter_comp_overlap(hidden_states, decoder_layer, tp_group, 0)
            
            # residual connection 1
            hidden_states = residual + hidden_states

            # residual 2
            residual = hidden_states

            # layernorm 2
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)

            # mlp1
            hidden_states = self.ring_all_gather_comp_overlap(hidden_states, decoder_layer, tp_group, 1)

            # activation： no activation, included in up_proj

            # mlp2
            if idx != len(self.layers) - 1:
                hidden_states = self.ring_reduce_scatter_comp_overlap(hidden_states, decoder_layer, 1)
                
                # residual connection 2
                hidden_states = residual + hidden_states
            
            else:
                # mlp2
                hidden_states = decoder_layer.down_proj(hidden_states)

                hidden_states[..., seq_l:seq_r, :] = residual + hidden_states[..., seq_l:seq_r, :]
                dist.reduce(hidden_states, dst=0)
            
        return hidden_states, next_decoder_cache, all_hidden_states, all_self_attns


    @torch.no_grad()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        galaxy=False,
        tp_group=None,
        prof=None,
    ):  
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).view(-1, seq_length)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            input_mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
            # print(f'rank: {self.tp_rank} start {self.vocab_start} end {self.vocab_end} input_mask: {input_mask}')
            masked_input = input_ids.clone() - self.vocab_start
            masked_input[input_mask] = 0
            inputs_embeds = self.embed_tokens(masked_input)
            # print(f'rank: {self.tp_rank} inputs_embeds: {inputs_embeds}')
            inputs_embeds[input_mask, :] = 0.0
            # print(f'rank: {self.tp_rank} after masked inputs_embeds: {inputs_embeds}')
            with prof.profile_context(f"Rank {dist.get_rank()}: all_reduce", device="cpu") if prof else nullcontext():
                dist.all_reduce(inputs_embeds, op=dist.ReduceOp.SUM, group=tp_group)
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds
        
        # if self.gradient_checkpointing and self.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #         )
        #         use_cache = False

        if not galaxy:
            hidden_states, next_decoder_cache, all_hidden_states, all_self_attns = self.forward_layers(
                hidden_states,
                past_key_values,
                position_ids,
                attention_mask,
                output_hidden_states,
                output_attentions,
                use_cache,
                tp_group=tp_group,
                prof=prof,
            )
        else:
            hidden_states, next_decoder_cache, all_hidden_states, all_self_attns = self.forward_layers_galaxy(
                hidden_states,
                past_key_values,
                position_ids,
                attention_mask,
                output_hidden_states,
                output_attentions,
                use_cache,
                tp_group=tp_group,
                prof=prof,
            )
                
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attns = () if output_attentions else None
        # next_decoder_cache = () if use_cache else None
        
        # for idx, decoder_layer in enumerate(self.layers):
        #     # print(f"rank: {self.tp_rank} start forward idx: {idx}")
        #     if output_hidden_states:
        #         all_hidden_states += (hidden_states,)
                
        #     past_key_value = (
        #         past_key_values[idx] if past_key_values is not None else None
        #     )
            
            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             # None for past_key_value
            #             return module(*inputs, output_attentions, None)

            #         return custom_forward

            #     layer_outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(decoder_layer),
            #         hidden_states,
            #         attention_mask,
            #         position_ids,
            #         None,
            #     )
            # else:
            #     # if idx == 0:
            #     #     if self.a == 0 or self.a == 1:
            #     #         print(f"stage {self.config.stage} hidden_states.shape: {hidden_states.shape}")
            #     #         print(f"stage {self.config.stage} hidden_states: {hidden_states}")
            #     #         print(f"stage {self.config.stage} attention_mask={attention_mask}")
            #     #         print(f"stage {self.config.stage} position_ids={position_ids}")
            #     #         print(f"stage {self.config.stage} past_key_value={past_key_value}")
            #     #         print(f"stage {self.config.stage} output_attentions={output_attentions}")
            #     #         print(f"stage {self.config.stage} use_cache={use_cache}")
            #     #         self.a += 1
                
            # layer_outputs = decoder_layer(
            #     hidden_states,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_value=past_key_value,
            #     output_attentions=output_attentions,
            #     use_cache=use_cache,
            #     tp_group=tp_group,
            #     prof=prof,
            # )
                
            # hidden_states = layer_outputs[0]

            # if use_cache:
            #     next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache
        )


class TPLlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, tp_rank: int = 0, tp_size: int = 1):
        super().__init__(config)
        self.model = TPLlamaModel(config, tp_rank=tp_rank, tp_size=tp_size)
        if tp_rank == 0:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
        # else:
        #     self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False,
        #                                         tp_rank=tp_rank, tp_size=tp_size)

    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    @torch.no_grad()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        tp_group=None,
        prof=None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            tp_group=tp_group,
            prof=prof,
        )

        # logits = self.lm_head(outputs.last_hidden_state)
        hidden_states = outputs[0]
        logits = None
        loss = None
        if self.config.stage == 0:
            logits = self.lm_head(hidden_states)
            logits = logits.float()
        else:
            return outputs
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # @classmethod
    # def from_tp_pretrained(cls, base_dir: str, tp_rank: int, tp_size: int, torch_dtype=torch.float16):
    #     stage_dir = None
    #     for name in os.listdir(base_dir):
    #         if name.startswith("stage_model_series_"):
    #             full_path = os.path.join(base_dir, name, f"stage_model_1", f"rank_{tp_rank}")
    #             if os.path.exists(os.path.join(full_path, "model.safetensors")):
    #                 stage_dir = full_path
    #                 break

    #     if stage_dir is None:
    #         raise ValueError(f"Cannot find rank_{tp_rank} in {base_dir}")

    #     config = LlamaConfig.from_pretrained(stage_dir)
    #     config.torch_dtype = torch_dtype

    #     model = cls(config, tp_rank=tp_rank, tp_size=tp_size)
    #     state_dict = load_file(os.path.join(stage_dir, "model.safetensors"))
    #     model.load_state_dict(state_dict, strict=True)
    #     model = model.cuda(tp_rank)
    #     model.eval()
    #     return model
