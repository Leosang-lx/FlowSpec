import json

import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
import os

from eagle.utils import *
from eagle.kv_cache import initialize_past_key_values
from eagle.cnets import Model
from eagle.configs import EConfig
from .stage_modeling_llama import StageLlamaModelForCausalLM
from .stage_ea_config import StageEaConfig
from .pipeline_utils import *


class StageEaModel(nn.Module):

    def __init__(
            self,
            stage_base_model,
            stage_base_model_or_path,
            ea_model_path,
            config,
            # config for drafting
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict=None  # draft model
    ):
        super().__init__()
        self.stage_base_model = stage_base_model
        self.config = stage_base_model.config
        self.hidden_size = stage_base_model.lm_head.weight.shape[-1]
        self.vocab_size = stage_base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = stage_base_model_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        # config = StageEaConfig.from_pretrained(ea_model_path)

        # [modify] model stage for tree decoding
        self.stage = self.config.stage
        self.total_stage = self.config.total_stage
        self.is_first_stage = self.stage == 0
        self.is_last_stage = self.stage == self.total_stage - 1


        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con['bias']
        except:
            bias = True

        # [modify] assume the draft ea_model is on the stage-0 device
        if config.is_first_stage:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k, threshold=threshold)
            low_memory = False
            assert isinstance(stage_base_model, StageLlamaModelForCausalLM)
            device = stage_base_model.model.partial_layers[-1].self_attn.q_proj.weight.device
            if device != stage_base_model.lm_head.weight.device:
                self.ea_layer.diff_device = True
                if not low_memory:
                    self.ea_layer.headweight = stage_base_model.lm_head.weight.clone().to(device)
                else:
                    self.ea_layer.layer_device = device

            else:
                self.ea_layer.diff_device = False
            self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
            self.ea_layer.to(self.stage_base_model.dtype).to(device)
            self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer  # maybe only the last stage need a tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            stage_base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        config = StageEaConfig.from_pretrained(ea_model_path)
        assert Type == 'LLaMA'  # only support LLaMA for now
        stage_base_model = StageLlamaModelForCausalLM.from_pretrained(
            stage_base_model_path, **kwargs
        )
        # Type = AutoConfig.from_pretrained(stage_base_model_path).architectures[0]

        config_path = os.path.join(ea_model_path, 'config.json')
        if not os.path.exists(config_path):
            config_path = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=stage_base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        stage_model = cls(
            stage_base_model,
            stage_base_model_path,
            config_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = stage_model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, stage_model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):  # warm up
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = stage_model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[
                times.index(min(times))]  # select the best total_size by testing the time/performance ratio
            stage_model.ea_layer.total_tokens = total_token - 1

        return stage_model

    @torch.no_grad()
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        outputs = self.stage_base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        if self.config.is_last_stage and output_orig:
            orig = self.stage_base_model.lm_head(hidden_states)
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate_pipeline(
            self,
            input_ids=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5 and self.config.is_last_stage:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        if self.config.is_first_stage:
            input_len = input_ids.shape[1]
            reset_tree_mode(self)
            # make a draft token tree
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor
            )
            new_token = 0

        for idx in range(max_length):
            # with Timer("all"):
            if self.config.is_first_stage:
                self.base_model.model.tree_mask = tree_mask
                draft_tokens = draft_tokens.to(input_ids.device)

            # todo: split token tree for pipelined verification
            logits, hidden_state_new, outputs = stage_tree_decoding()




