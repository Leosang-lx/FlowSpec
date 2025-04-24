""" PyTorch LLaMA model."""
from typing import Optional, Tuple, Union

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
from eagle.modeling_llama_kv import LlamaPreTrainedModel, LlamaRMSNorm  # LlamaDecoderLayer
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from eagle.modeling_llama_kv import _expand_mask, _make_causal_mask, LlamaDecoderLayer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class StageLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, embed_tokens=None, hidden_layers=None, post_init=True):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False
        # self.a = 0
        # [modify] is_first_stage and is_last_stage in config is identified by the stage number
        self.config = config
        if config.has_embedding:
            if embed_tokens is not None and isinstance(embed_tokens, nn.Embedding):
                # print(f"embed_tokens: {embed_tokens}")
                self.embed_tokens = embed_tokens
            else:
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # [modify] the stage model only have partial layers, defined by layer_range = [start_idx, end_idx)
        if hidden_layers is not None:
            self.layers = hidden_layers
        else:
            self.layers = nn.ModuleList(
                [LlamaDecoderLayer(config) for layer_idx in range(*config.layer_range)]
            )
        if config.is_last_stage:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.has_lm_head and not config.has_embedding:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights adn apply final processing
        if post_init:
            self.post_init()

    def get_input_embeddings(self):
        # [modified]
        if self.config.is_first_stage:
            return self.embed_tokens
        else:
            return None

    def set_input_embeddings(self, value):
        # [modified]
        if self.config.is_first_stage:
            self.embed_tokens = value
        else:
            pass

    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
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
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_tgt_len = tree_mask.size(-2)
            tree_src_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_tgt_len:, -tree_src_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()
            # if self.config.is_first_stage:
            #     if self.a == 0 or self.a == 1:
            #         print(f"combined_attention_mask: {combined_attention_mask}")
            #         self.a += 1
        return combined_attention_mask

    @torch.no_grad()
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,  # [MODIFIED] past_key_value is KVCache class
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # embed positions
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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        # if self.config.is_first_stage:  
        #     if self.a == 0 or self.a == 1 or self.a == 2 or self.a == 3 or self.a == 4 or self.a == 5:
        #         print(f"len_past_key_values: {len(past_key_values)}")
        #         print(f"len_past_key_values[0]: {len(past_key_values[0])}")
        #         print(f"past_key_values[0][0].shape: {past_key_values[0][0].shape}")
        #         self.a += 1
        for idx, decoder_layer in enumerate(self.layers):
            # if idx==16:
            #     print(idx)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                # if idx == 0:
                #     if self.a == 0 or self.a == 1:
                #         print(f"stage {self.config.stage} hidden_states.shape: {hidden_states.shape}")
                #         print(f"stage {self.config.stage} hidden_states: {hidden_states}")
                #         print(f"stage {self.config.stage} attention_mask={attention_mask}")
                #         print(f"stage {self.config.stage} position_ids={position_ids}")
                #         print(f"stage {self.config.stage} past_key_value={past_key_value}")
                #         print(f"stage {self.config.stage} output_attentions={output_attentions}")
                #         print(f"stage {self.config.stage} use_cache={use_cache}")
                #         self.a += 1
                
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

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.config.is_last_stage:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class StageLlamaModelForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    # [update] init with lm_head
    def __init__(self, config, stage_model=None, lm_head=None):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        if stage_model is not None and isinstance(stage_model, StageLlamaModel):
            self.model = stage_model
        else:
            self.model = StageLlamaModel(config)
            
        if config.has_lm_head:
            if config.has_embedding:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                self.lm_head.weight = self.model.embed_tokens.weight
            else:
                assert self.model.lm_head is not None
                self.lm_head = self.model.lm_head

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # [modify]
        if self.config.is_first_stage:
            return self.model.embed_tokens
        else:
            return None

    def set_input_embeddings(self, value):
        # [modify]
        if self.config.is_first_stage:
            self.model.embed_tokens = value

    def get_output_embeddings(self):
        # [modify]
        if self.config.is_last_stage:
            return self.lm_head
        else:
            return None

    def set_output_embeddings(self, new_embeddings):
        # [modify]
        if self.config.is_last_stage:
            self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    @torch.no_grad()
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,  # [MODIFIED] past_key_value is KVCache class
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        )
        # [modify] return outputs: BaseModelOutputWithPast if not the last stage
        if not self.config.is_last_stage:
            return outputs
        # if output_orig:
        #     orig = self.lm_head(outputs[0])

        hidden_states = outputs[0]
        # if self.pretraining_tp > 1:
        #     lm_head_slices = self.lm_head.weight.split(
        #         self.vocab_size // self.pretraining_tp, dim=0
        #     )
        #     logits = [
        #         F.linear(hidden_states, lm_head_slices[i])
        #         for i in range(self.pretraining_tp)
        #     ]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


