import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
import os

from eagle.utils import *
from eagle.kv_cache import initialize_past_key_values
from eagle.cnets import Model
# from eagle.configs import EConfig
from stage_modeling_llama import StageLlamaModelForCausalLM
from stage_ea_config import StageEaConfig
from pipeline_utils import *


class StageEaModel(nn.Module):

    def __init__(
            self,
            stage_base_model,
            stage_base_model_or_path,
            # ea_model_path,
            config,
            # total_token,
            # depth,
            # top_k,
            # threshold,
            # ea_layer_state_dict=None,
            ea_draft_model=None  # draft model
    ):
        super().__init__()
        self.stage_base_model = stage_base_model
        self.config = stage_base_model.config
        if config.has_lm_head:
            self.hidden_size = stage_base_model.lm_head.weight.shape[-1]
            self.vocab_size = stage_base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = stage_base_model_or_path
        if config.is_first_stage:  # has embedding and lm_head
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        else:
            self.tokenizer = None
        # config = StageEaConfig.from_pretrained(ea_model_path)

        # [MODIFIED] stage model for pipelined tree decoding
        self.stage = self.config.stage
        self.total_stage = self.config.total_stage
        self.is_first_stage = self.stage == 0
        self.is_last_stage = self.stage == self.total_stage - 1

        # [MODIFIED] assume the draft ea_model is on the stage-0 device
        if config.has_draft_model:
            assert ea_draft_model is not None
            assert isinstance(ea_draft_model, Model)
            self.ea_layer = ea_draft_model
            self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer  # maybe only the first stage need a tokenizer

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
        model_config = StageEaConfig.from_pretrained(stage_base_model_path)
        assert Type == 'LLaMA'  # only support LLaMA for now
        stage_base_model = StageLlamaModelForCausalLM.from_pretrained(
            stage_base_model_path, **kwargs
        )
        # if model_config.has_lm_head:
        #     print(f"stage_base_model.lm_head.weight.device={stage_base_model.lm_head.weight.device}")
        # Type = AutoConfig.from_pretrained(stage_base_model_path).architectures[0]

        # [MODIFIED] load draft model when config.has_draft_model==True
        if model_config.has_draft_model:
            assert ea_model_path is not None
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

            # [MODIFIED] load ea_draft_model in from_pretrained()
            ea_config = StageEaConfig.from_pretrained(config_path)
            # load the draft model
            with open(config_path, "r") as f:
                con = json.loads(f.read())
            try:
                bias = con['bias']
            except:
                bias = True

            ea_layer = Model(ea_config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                             threshold=threshold)
            low_memory = False

            device = stage_base_model.model.layers[-1].self_attn.q_proj.weight.device
            # print(f"stage_base_model.lm_head.weight.device={stage_base_model.lm_head.weight.device}")
            # print(f"device={device}")
            if device != stage_base_model.lm_head.weight.device:
                ea_layer.diff_device = True
                if not low_memory:
                    ea_layer.headweight = stage_base_model.lm_head.weight.clone().to(device)
                else:
                    ea_layer.layer_device = device

            else:
                ea_layer.diff_device = False
            ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
            ea_layer.to(stage_base_model.dtype).to(device)
            # print(f"ea_layer.embed_tokens.weight.device={ea_layer.embed_tokens.weight.device}")
        else:
            ea_layer = None

        stage_model = cls(
            stage_base_model,
            stage_base_model_path,
            model_config,
            ea_layer
            # total_token,
            # depth,
            # top_k,
            # threshold,
            # ea_layer_state_dict
        )

        if total_token == -1:
            raise NotImplementedError("total_token == -1 is not implemented")
            device = stage_model.stage_base_model.model.layers[0].self_attn.q_proj.weight.device
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
                        # TODO: set the pipeline frame here
                        outputs = stage_model.stage_base_model(input_ids)
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
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):  
            
        if self.is_first_stage:
            outputs = self.stage_base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids
            )
        else:
            outputs = self.stage_base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids
            )
            
        hidden_states = outputs[0]
        
        if self.is_last_stage and output_orig:
            orig = self.stage_base_model.lm_head(hidden_states)
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states   

    @torch.no_grad()  # collaborative function
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
        """
        pipelined tree decoding
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.is_first_stage:
            max_length=max_length-self.ea_layer.total_tokens-10
        # [MODIFIED]: only first stage and last stage has the ea_layer, embedding, lm_head and input_ids
        if temperature > 1e-5 and (self.is_first_stage or self.is_last_stage):
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

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
            ) = initialize_past_key_values(self.stage_base_model)  # todo: init cache with split layer_num
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        
        # to determine the stop of the pipeline
        should_stop = torch.tensor(0, dtype=torch.int32, device=self.stage_base_model.device)
        
        # print(f"stage {self.stage} barrier")
        # dist.barrier()
        
        if self.is_first_stage:
            max_length = max_length - self.ea_layer.total_tokens - 10
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoid modifying the input_ids in-place
            # padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

            input_len = input_ids.shape[1]
            reset_tree_mode(self)
            # make a draft token tree
            # print(f"rank={self.stage}, self.ea_layer.embed_tokens.weight.device={self.ea_layer.embed_tokens.weight.device}")
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree_pipeline(
                self, past_key_values, logits_processor, input_ids
            )
            retrieve_indices = retrieve_indices.to(self.stage_base_model.device)
            # split the tree for pipeline
            
            # print(f"tree_mask: {tree_mask}")
            # print(f"tree_mask_split: {tree_mask_split}")
            # print(f"draft_tokens: {draft_tokens}")
            # print(f"seqs_split: {seqs_split}")
            # print(f"tree_position_ids: {tree_position_ids}")
            # print(f"tree_position_ids.dtype: {tree_position_ids.dtype}")
            # print(f"tree_pos_ids_split: {tree_pos_ids_split}")
            # Tuple(subtree1=(draft_tokens, tree_position_ids, tree_attention_mask, retrieve_indices), ...)
            new_token = 0
        else:
            initialize_tree_pipeline(self, past_key_values)
        
        # print(f"stage {self.stage} barrier")
        # dist.barrier()
        for idx in range(max_length):
            
            
            # if self.is_first_stage:
            #     # send mask to all stages
            #     mask_shape = torch.tensor(tree_mask.shape, dtype=torch.int32)
            #     self.base_model.model.tree_mask = tree_mask
            #     dist.broadcast(mask_shape, src=0)
            #     dist.broadcast(tree_mask, src=0)
            
            # else:
            #     # todo: maybe end the iterations in here for other stages?
            #     mask_shape = torch.zeros(4, dtype=torch.int32)  # shape like (1, 1, seq_len, seq_len)
            #     dist.broadcast(mask_shape, src=0)
            #     tree_mask = torch.zeros(mask_shape, dtype=torch.float64)
            #     dist.broadcast(tree_mask, src=0)
            
            # [tree_decoding]
            
            if self.is_first_stage:
                seqs_split, tree_pos_ids_split, tree_mask_split, lens_split = tree_partition_pipeline(
                    draft_tokens,
                    tree_position_ids,
                    tree_mask, 
                    self.total_stage
                )
                # config tree mask for each stage in stage_tree_decoding()
                # self.stage_base_model.model.tree_mask = tree_mask
                tree_decoding_params = (
                self, past_key_values, retrieve_indices, seqs_split, tree_pos_ids_split, lens_split, input_ids, tree_mask_split)
            else:
                tree_decoding_params = (self, past_key_values)

            outputs = stage_tree_decoding(*tree_decoding_params)

            # print(f"stage {self.stage} barrier waiting for tree decoding")
            # dist.barrier()
            if self.is_first_stage:
                padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(self.stage_base_model.device)
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                send(draft_tokens, dst=self.total_stage-1)
                send(retrieve_indices, dst=self.total_stage-1)
            # [evaluate_posterior]
            if self.is_last_stage:
                draft_tokens = recv(src=0, data_type=torch.int64, shape_length=2).to(self.stage_base_model.device)
                retrieve_indices = recv(src=0, data_type=torch.int64, shape_length=2).to(self.stage_base_model.device)
                
                # padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(self.stage_base_model.device)
                # draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]
                logits, hidden_state = outputs  # get the outputs of tree decoding
                # if idx == 0:    
                #     print(f"logits: {logits}")
                #     print(f"candidates: {candidates}")
                #     print(f"hidden_state: {hidden_state}")
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
                # if idx == 0:
                #     print(f"best_candidate: {best_candidate}")
                #     print(f"accept_length: {accept_length}")
                #     print(f"sample_p: {sample_p}")
                # # should be optimized
                send(hidden_state, dst=0)
                send(best_candidate, dst=0)
                accept_length = torch.tensor(accept_length, dtype=torch.int64)
                send(accept_length, dst=0)
                send(sample_p, dst=0)
                
            if self.is_first_stage:
                hidden_state = recv(src=self.total_stage-1, data_type=torch.float16, shape_length=3).to(self.stage_base_model.device)
                # should be optimized
                best_candidate = recv(src=self.total_stage-1, data_type=torch.int64, shape_length=0).to(self.stage_base_model.device)
                accept_length = recv(src=self.total_stage-1, data_type=torch.int64, shape_length=0).to(self.stage_base_model.device)
                sample_p = recv(src=self.total_stage-1, data_type=torch.float16, shape_length=1).to(self.stage_base_model.device)
                
                # print(f"hidden_state: {hidden_state}")
                # print(f"best_candidate: {best_candidate}")
                # print(f"accept_length: {accept_length}")
                # print(f"sample_p: {sample_p}")
            # [update_inference_inputs]
            """
            OWNED
            stage_1: input_ids, prev_len, candidates
            middle_stage: 
            stage_n-1: logits, candidates, retrieve_indices, best_candidates, accept_length, select_indices
            NEED
            
            """
            # print(f"stage {self.stage} barrier waiting for update_inference_inputs")
            # dist.barrier()
            if self.is_first_stage:
                candidates = draft_tokens[0, retrieve_indices]
                update_inputs_params = (
                    self,
                    past_key_values_data,
                    current_length_data,
                    logits_processor,
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    new_token,
                    hidden_state,
                    sample_p,
                )
            else:
                update_inputs_params = (self, past_key_values_data, current_length_data)

            outputs = update_stage_inference_inputs(*update_inputs_params)
            
            # print(f"stage {self.stage} barrier waiting for check stop")
            # dist.barrier()
            if self.is_first_stage:
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = outputs

                if input_ids is not None and self.tokenizer is not None:
                    if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif new_token > max_new_tokens:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif input_ids.shape[1] > max_length:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    
                    broadcast(should_stop, src=0)
                    if should_stop.item() == 1:
                        break
            else:
                should_stop = broadcast(src=0, data_type=torch.int32, shape_length=0)
                if should_stop.item():
                    break
        if self.is_first_stage:
            if not log:
                return input_ids
            else:
                return input_ids, new_token, idx

    @torch.no_grad()  # collaborative function
    def eagenerate_pruned_pipeline(
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

        # [MODIFIED]: only stage1 has the ea_layer and input_ids
        if temperature > 1e-5 and self.config.is_last_stage:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

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
            ) = initialize_past_key_values(self.base_model)  # todo: init cache with split layer_num
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        # [initialization]
        config = self.config
        if config.is_first_stage or config.is_last_stage:
            padding = torch.zeros(1, 1, dtype=torch.long) - 1  # padding -1 to the draft token sequence

        # [initialization] model
        if config.is_first_stage:
            max_length = max_length - self.ea_layer.total_tokens - 10
            assert input_ids is not None, '"input_ids" cannot be None for stage1'
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoiding modifying the input_ids in-place
            padding = padding.to(input_ids.device)
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

            input_len = input_ids.shape[1]
            reset_tree_mode(self)

        # [initialization] prefill of base model
        if config.is_first_stage:
            assert input_ids is not None
        output = prefill_pipeline(
            self, input_ids=input_ids, stage_past_key_values=past_key_values
        )
        if config.is_first_stage:
            orig, hidden_state, past_key_values = output
        else:
            past_key_values = output

        # get the global accept length (for later pruning)
        global_accept_len = past_key_values[0][0].size(-2)

        for idx_spec in range(max_length):

            if config.is_first_stage:
                token = process_logits(orig[:, -1], logits_processor)
                input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

                # make a draft token tree based on the hidden_state and input_ids
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
                    hidden_state, input_ids_ea, self.base_model.lm_head,
                    logits_processor)  # todo: stage_model1 has no lm_head

                # split the tree
                draft_tokens_split, lens_split, tree_position_ids_split, subseq_ri_cum_depths = token_tree_partition(
                    draft_tokens, retrieve_indices, tree_position_ids, config.total_stage)

            else:
                lens_split = torch.zeros(config.total_stage, dtype=torch.long)
            dist.broadcast(lens_split, src=0)
            # n_subseq = lens_split.size(-1)  # todo: may unequal to config.total_stage
            draft_len = torch.sum(lens_split, dim=0)

            # tree synchronization
            if not config.is_first_stage:
                mask_shape = draft_len.repeat(0)
                tree_mask = torch.zeros(mask_shape, dtype=torch.long)[None, None]
            # send tree_mask to all stages
            dist.broadcast(tree_mask, src=0)
            self.stage_base_model.model.tree_mask = tree_mask
            # send draft token sequence and retrieve_indices
            if config.is_first_stage:
                dist.send(draft_tokens, dst=config.last_rank)
                ri_shape = torch.tensor(retrieve_indices.shape, dtype=torch.long)

                dist.send(ri_shape, dst=config.last_rank)
                dist.send(retrieve_indices, dst=config.last_rank)
                dist.send(subseq_ri_cum_depths, dst=config.last_rank)
            if config.is_last_stage:
                # draft_tokens
                draft_tokens = torch.zeros(1, draft_len, dtype=torch.long)
                dist.recv(draft_tokens, src=0)
                # retrieve_indices
                ri_shape = torch.zeros(2, dtype=torch.long)
                dist.recv(ri_shape, src=0)
                retrieve_indices = torch.zeros(ri_shape, dtype=torch.long)
                dist.recv(retrieve_indices)
                subseq_ri_cum_depths = torch.zeros(ri_shape[0].item(), config.total_stage, dtype=torch.long)

            # fill the pipeline stages
            # - stage1 do not need to recv anything in this step
            cum_lens = torch.cumsum(lens_split)
            isend_task = None
            for i in range(config.total_stage - config.stage):
                if config.is_first_stage:
                    subseq_ids = draft_tokens_split[i]
                    subseq_pos_ids = tree_position_ids_split[i]
                    # cum_len += subseq_ids.size(-1)

                    sub_hidden_state, past_key_values = self.stage_base_model(
                        input_ids=subseq_ids,
                        past_key_values=past_key_values,
                        position_ids=subseq_pos_ids,
                        tree_mask_range=cum_lens[i]
                    )
                else:
                    last_hidden_state = torch.zeros((1, lens_split[i], config.hidden_size),
                                                    dtype=torch.float16)  # todo: d_model?
                    dist.recv(last_hidden_state, src=config.last_rank)
                    sub_hidden_state, past_key_values = self.stage_base_model.model(
                        last_hidden_state=last_hidden_state,
                        past_key_values=past_key_values,
                        tree_mask_range=cum_lens[i]
                    )
                if isend_task is not None:
                    isend_task.wait()
                # keep the last hidden_state after all pipeline stages are filled
                if i < config.total_stage - config.stage - 1:
                    isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)

            if self.is_first_stage:
                accept_hidden_states = []

            # continuous verification and pruning
            # 从stage4计算完他的hidden_state开始inner loop
            for i in range(config.total_stage):
                # pruning
                if config.is_last_stage:
                    # [verification]
                    subtree_logits = self.stage_base_model.lm_head(sub_hidden_state)
                    draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                    sub_retrieve_indices = get_subtree_retrieve_indices(retrieve_indices, subseq_ri_cum_depths[0])
                    candidates = draft_tokens[0, sub_retrieve_indices]
                    best_candidate, accept_length, sample_p = evaluate_posterior(
                        subtree_logits, candidates, logits_processor
                    )
                    global_accept_len += accept_length
                    new_token = process_logits(sample_p, logits_processor)

                    # [local pruning]
                    output = pruning(draft_tokens, retrieve_indices, best_candidate, accept_length, new_token,
                                     subseq_ri_cum_depths)
                    if output is None:  # start new speculation round
                        dist.broadcast(torch.zeros(1, dtype=torch.long))
                        break

                    retrieve_indices, left_indices, subseq_ri_cum_depths = output
                    left_hs_indices = left_indices[left_indices < sub_hidden_state.size(-2)]
                    left_hidden_state = sub_hidden_state[..., left_hs_indices, :]
                    dist.broadcast(left_indices.shape, src=config.stage)

                    pruning_info = torch.cat((accept_length, left_indices), dim=0).contiguous()
                    dist.broadcast(pruning_info, src=config.stage)  # maybe use dist.send() instead
                    dist.send(left_hidden_state, dst=config.next_rank)

                else:
                    # if continue
                    pruning_info_shape = torch.zeros(1, dtype=torch.long)
                    dist.broadcast(pruning_info_shape, src=config.total_stage - 1)
                    if pruning_info_shape == 0:  # 0 means new round of speculative decoding
                        break
                    pruning_info = torch.zeros(pruning_info_shape, dtype=torch.long)
                    dist.broadcast(pruning_info, dsrc=config.total_stage - 1)
                    accept_length = pruning_info[0].item()
                    left_indices = pruning_info[1:]
                    global_accept_len += accept_length

                    if config.is_first_stage:
                        left_hidden_state = torch.zeros(1, accept_length, config.d_model, dtype=torch.float16)
                        irecv_task = dist.irecv(left_hidden_state, src=config.last_rank)
                        left_draft_tokens, left_tree_pos_ids, transformed_ri, accepted_tokens, new_token = stage1_pruning(
                            left_indices, accept_length, draft_tokens, retrieve_indices, tree_position_ids
                        )
                        input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                        irecv_task.wait()
                        accept_hidden_states.append(left_hidden_state)

                # [global pruning] prune the tokens (kv_cache and hidden_state)
                # - according to the global_accept_len and the hidden_state_len
                past_key_values_data, current_length_data, sub_hidden_state, tree_mask = token_pruning(
                    past_key_values_data,
                    current_length_data,
                    self.stage_base_model.model.tree_mask,
                    sub_hidden_state,
                    left_indices,
                    global_accept_len,
                    accept_length
                )
                # 这么写总感觉哪里怪怪的
                if i < config.total_stage:
                    if isend_task is not None:
                        isend_task.wait()
                    isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)
                    dist.recv(last_hidden_state, src=config.last_rank)
                    sub_hidden_state, past_key_values = self.stage_base_model.model(
                        last_hidden_state=last_hidden_state,
                        past_key_values=past_key_values,
                        tree_mask_range=cum_lens[i]
                    )

            if config.is_first_stage:
                # stop criteria
                if input_ids is not None and self.tokenizer is not None:
                    if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif new_token > max_new_tokens:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)
                    elif input_ids.shape[1] > max_length:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.stage_base_model.device)

                    dist.broadcast(should_stop, src=0)
                    if should_stop.item() == 1:
                        break
                else:
                    dist.broadcast(should_stop, src=0)
                    if should_stop.item():
                        break
                hidden_state = torch.cat(accept_hidden_states, dim=-2)  # for draft generation next round
        if self.is_first_stage:
            if not log:
                return input_ids
            else:
                return input_ids, new_token, idx_spec

    @torch.no_grad()  # collaborative function
    def eagenerate_continuous(
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

        # [MODIFIED]: only stage1 has the ea_layer and input_ids
        if temperature > 1e-5 and self.config.is_last_stage:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # initialize hte past key and value states
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
            ) = initialize_past_key_values(self.base_model)  # todo: init cache with split layer_num
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        # [initialization]
        config = self.config
        if config.is_first_stage or config.is_last_stage:
            padding = torch.zeros(1, 1, dtype=torch.long) - 1  # padding -1 to the draft token sequence

        # [initialization] model
        if config.is_first_stage:
            max_length = max_length - self.ea_layer.total_tokens - 10
            assert input_ids is not None, '"input_ids" cannot be None for stage1'
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoiding modifying the input_ids in-place
            padding = padding.to(input_ids.device)
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

            input_len = input_ids.shape[1]
            reset_tree_mode(self)

        # [initialization] prefill of base model
        if config.is_first_stage:
            assert input_ids is not None
        output = prefill_pipeline(
            self, input_ids=input_ids, stage_past_key_values=past_key_values
        )
        if config.is_first_stage:
            orig, hidden_state, past_key_values = output
        else:
            past_key_values = output

        global_accept_len = past_key_values[0][0].size(-2)

        # [continuous speculation] outer loop: start from a new draft tree
        for idx_spec in range(max_length):

            if config.is_first_stage:
                token = process_logits(orig[:, -1], logits_processor)
                input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

                # make a draft token tree based on the hidden_state and input_ids
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
                    hidden_state, input_ids_ea, self.base_model.lm_head,
                    logits_processor)  # todo: stage_model1 has no lm_head

                # split the tree
                draft_tokens_split, lens_split, tree_position_ids_split, subseq_ri_cum_depths = token_tree_partition(
                    draft_tokens, retrieve_indices, tree_position_ids, config.total_stage)

            else:
                lens_split = torch.zeros(config.total_stage, dtype=torch.long)
            dist.broadcast(lens_split, src=0)

            draft_len = int(torch.sum(lens_split, dim=0))
            # tree synchronization
            if not config.is_first_stage:
                mask_shape = draft_len.repeat(0)
                tree_mask = torch.zeros(mask_shape, dtype=torch.long)[None, None]
            # send tree_mask to all stages
            dist.broadcast(tree_mask, src=0)
            self.stage_base_model.model.tree_mask = tree_mask
            # send draft token sequence and retrieve_indices
            if config.is_first_stage:
                dist.send(draft_tokens, dst=config.last_rank)
                ri_shape = torch.tensor(retrieve_indices.shape, dtype=torch.long)

                dist.send(ri_shape, dst=config.last_rank)
                dist.send(retrieve_indices, dst=config.last_rank)
                dist.send(subseq_ri_cum_depths, dst=config.last_rank)
            if config.is_last_stage:
                # draft_tokens
                draft_tokens = torch.zeros(1, draft_len, dtype=torch.long)
                dist.recv(draft_tokens, src=0)
                # retrieve_indices
                ri_shape = torch.zeros(2, dtype=torch.long)
                dist.recv(ri_shape, src=0)
                retrieve_indices = torch.zeros(ri_shape, dtype=torch.long)
                dist.recv(retrieve_indices)
                subseq_ri_cum_depths = torch.zeros(ri_shape[0].item(), config.total_stage, dtype=torch.long)

            # fill the pipeline stages
            # - stage1 do not need to recv anything in this step
            cum_lens = torch.cumsum(lens_split)
            for i in range(config.total_stage - config.stage):
                isend_task = None
                if config.is_first_stage:
                    subseq_ids = draft_tokens_split[i]
                    subseq_pos_ids = tree_position_ids_split[i]
                    # cum_len += subseq_ids.size(-1)

                    # todo: return the past_key_values from the base model
                    sub_hidden_state, past_key_values = self.stage_base_model(
                        input_ids=subseq_ids,
                        past_key_values=past_key_values,
                        position_ids=subseq_pos_ids,
                        tree_mask_range=cum_lens[i]
                    )
                else:
                    last_hidden_state = torch.zeros((1, lens_split[i], config.d_model),
                                                    dtype=torch.float16)  # todo: d_model?
                    dist.recv(last_hidden_state, src=config.last_rank)
                    sub_hidden_state, past_key_values = self.stage_base_model.model(
                        last_hidden_state=last_hidden_state,
                        past_key_values=past_key_values,
                        tree_mask_range=cum_lens[i]
                    )
                if isend_task is not None:
                    isend_task.wait()
                # keep the last hidden_state after all pipeline stages are filled
                if i != config.total_stage - config.stage:
                    isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)

            # [continuous pipeline speculation]
            while True:
                if config.is_last_stage:
                    # [verification]
                    subtree_logits = self.stage_base_model.lm_head(sub_hidden_state)
                    draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                    sub_retrieve_indices = get_subtree_retrieve_indices(retrieve_indices, subseq_ri_cum_depths[0])
                    candidates = draft_tokens[0, sub_retrieve_indices]
                    best_candidate, accept_length, sample_p = evaluate_posterior(
                        subtree_logits, candidates, logits_processor
                    )
                    token = process_logits(sample_p, logits_processor)

                    # [local pruning]
                    output = pruning(
                        draft_tokens, retrieve_indices, best_candidate, accept_length, token, subseq_ri_cum_depths
                    )
                    if output is None:  # start new speculation round
                        pass  # todo: tell all stages to start new round
                        break

                    retrieve_indices, left_indices, subseq_ri_cum_depths = output
                    left_hs_indices = left_indices[left_indices < sub_hidden_state.size(-2)]
                    left_hidden_state = sub_hidden_state[..., left_hs_indices, :]
                    dist.broadcast(left_indices.shape, src=config.stage)
                    pruning_info = torch.cat((accept_length, left_indices), dim=0).contiguous()
                    dist.broadcast(pruning_info, src=config.stage)  # maybe use dist.send() instead
                    # todo: send the accepted_hidden_state to stage1 for draft generation
                    dist.send(left_hidden_state, dst=config.next_rank)

                else:
                    # todo: receive info from the last stage to check whether continue the continuous speculation
                    # if continue
                    left_size = torch.zeros(1, dtype=torch.long)
                    dist.broadcast(left_size, src=config.total_stage - 1)
                    pruning_info = torch.zeros(left_size + 1, dtype=torch.long)
                    dist.broadcast(pruning_info, src=config.total_stage - 1)
                    accept_length = pruning_info[0].item()
                    left_indices = pruning_info[1:]
                    if config.is_first_stage:
                        left_hidden_state = torch.zeros(1, accept_length, config.d_model, dtype=torch.float16)
                        irecv_task = dist.irecv(left_hidden_state, src=config.last_rank)
                        left_draft_tokens, left_tree_pos_ids, transformed_ri, accepted_tokens, new_token = stage1_pruning(
                            left_indices, accept_length, draft_tokens, retrieve_indices, tree_position_ids
                        )
                        input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                        irecv_task.wait()

                # [global pruning] prune the tokens (kv_cache and hidden_state)
                # - according to the global_accept_len and the hidden_state_len
                past_key_values_data, current_length_data, hidden_state, tree_mask = token_pruning(
                    past_key_values_data, current_length_data, tree_mask, hidden_state, left_indices, global_accept_len,
                    accept_length
                )
                global_accept_len += accept_length

                if not config.is_last_stage:
                    if isend_task is not None:
                        isend_task.wait()
                    isend_task = dist.isend(hidden_state, dst=config.next_rank)
                if not config.is_first_stage:
                    dist.recv(hidden_state, src=config.last_rank)
                    cum_hs_len = current_length_data[0] + hidden_state.size(-2)
                    sub_hidden_state, past_key_values = self.stage_base_model.model(
                        last_hidden_state=last_hidden_state,
                        past_key_values=past_key_values,
                        tree_mask_range=cum_hs_len
                    )
                else:  # the draft tree expand in the first stage
                    grow_token_tree(
                        self,
                        torch.cat((input_ids, new_token), dim=-1),
                        left_hidden_state,
                        left_draft_tokens,

                    )
