import json
# from memory_profiler import profile
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
import os
import gc

from eagle.utils import *
from eagle.kv_cache import initialize_past_key_values
from eagle.cnets import Model
# from eagle.configs import EConfig
from model.stage_modeling_llama import StageLlamaModelForCausalLM
from stage_ea_config import StageEaConfig
from pipeline_utils import *
from comm.comm_handler import CommHandler
# from torch.nn.init import init_empty_weights
from accelerate import init_empty_weights
from contextlib import nullcontext
from profiler.profiler import prof
from tools.length_sweep import length_sweep
from config.run_config import config as run_config
from datetime import timedelta

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
            ea_draft_model=None,  # draft model
            init_comm=True
    ):
        super().__init__()
        self.stage_base_model = stage_base_model
        self.config = stage_base_model.config
        if config.has_lm_head:
            self.hidden_size = stage_base_model.lm_head.weight.shape[-1]
            self.vocab_size = stage_base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = stage_base_model_or_path
        if config.is_first_stage or config.is_draft_stage:  # has embedding and lm_head
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        else:
            self.tokenizer = None
        # config = StageEaConfig.from_pretrained(ea_model_path)

        # [MODIFIED] stage model for pipelined tree decoding
        self.stage = self.config.stage
        self.total_stage = self.config.total_stage
        # [update]: add is_draft_stage
        self.is_draft_stage = self.stage == 0
        self.is_first_stage = self.stage == 1
        self.is_last_stage = self.stage == self.total_stage - 1

        # [MODIFIED] assume the draft ea_model is on the stage-0 device
        if config.has_draft_model:
            assert ea_draft_model is not None
            assert isinstance(ea_draft_model, Model)
            self.ea_layer = ea_draft_model
            self.ea_layer.init_tree()

        # [MODIFIED] initialize comm handler
        # print(f"start init comm handler")
        if init_comm:
            self.comm = CommHandler(rank=config.stage, world_size=config.total_stage, timeout=run_config.timeout)
            # print(f"init comm handler")
            self.comm.init_PG()
            self.comm.start_threads()
            if run_config.hardware == "jetson" and not run_config.set_network:
                self.comm.reset_traffic()       
            if run_config.hardware == "jetson" and run_config.set_network:
                self.comm.traffic_control(run_config.rate_mbps, run_config.delay_ms)
            dist.barrier()
        
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
            init_comm=True,
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
        if model_config.is_first_stage and total_token == -1:
            # print(f"length_sweep(stage_base_model) * model_config.total_stage={length_sweep(stage_base_model) * model_config.total_stage}")
            total_token = length_sweep(stage_base_model) * model_config.total_stage
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
                ea_layer_state_dict = load_file(load_model_path,
                                                device=stage_base_model.device)
                
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

            # device = stage_base_model.model.layers[-1].self_attn.q_proj.weight.device
            device = stage_base_model.lm_head.weight.device
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
            
            del ea_layer_state_dict, ea_config, con
            torch.cuda.empty_cache()
            
            ea_layer.to(stage_base_model.dtype).to(device)
            # print(f"ea_layer.embed_tokens.weight.device={ea_layer.embed_tokens.weight.device}")
        else:
            ea_layer = None

        stage_model = cls(
            stage_base_model,
            stage_base_model_path,
            model_config,
            ea_layer,
            # total_token,
            # depth,
            # top_k,
            # threshold,
            # ea_layer_state_dict
            init_comm=init_comm
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
        # the draft stage will not call forward()    
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
        
    def draft_init_pipedec(
        self,
        stage_past_key_values=None,
        input_ids=None,  # including the sampled token
        token=None,
        hidden_state=None,
        logits_processor=None,
        log=True,
        prof=None,
    ):
        config = self.config
        device = self.stage_base_model.device
        comm = self.comm

        if config.is_draft_stage:
            input_ids_ea = torch.cat((input_ids, token), dim=1)

            lens_split = []
            # draft_tokens = input_ids[:, -1][None]
            # print(f'draft_tokens: {self.tokenizer.decode(draft_tokens[0])}')
            draft_tokens = token
            
            tree_position_ids = torch.zeros(1, dtype=torch.long) + input_ids.size(-1)
            tree_mask = torch.ones(1, 1, 1, 1, dtype=torch.float32)
            retrieve_indices = torch.zeros(1, 1, dtype=torch.long)
            for i in range(config.total_stage):
                # with open('draft_tokens.txt', 'a') as f:
                #     f.write(f'{self.tokenizer.decode(draft_tokens[0])}\n')
                if i == 0:
                    appended_input = draft_tokens
                    appended_tree_pos_ids = tree_position_ids
                    appended_tree_mask = tree_mask
                elif i == 1:
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.expand_pipedec(
                            hidden_state,
                            input_ids_ea,  # input_ids_ea只有first expand时候要用
                            self.stage_base_model.lm_head,
                            logits_processor,
                            top_k=run_config.init_topk_pipedec,
                            first_expand=True
                            # prof=prof
                    )
                    tree_position_ids = tree_position_ids + input_ids.size(-1)
                    appended_input = draft_tokens[:, 1:]
                    appended_tree_pos_ids = tree_position_ids[1:]
                    appended_tree_mask = tree_mask[:, :, 1:, :]
                    
                else:
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.expand_pipedec(
                        None,
                        input_ids,
                        self.stage_base_model.lm_head,
                        logits_processor,
                        top_k=run_config.init_topk_pipedec,
                        last_state=last_ea_state,
                        tree=last_ea_tree,
                        first_expand=False,
                    )
                    appended_input = draft_tokens[:, -run_config.init_topk_pipedec:]
                    appended_tree_pos_ids = tree_position_ids[-run_config.init_topk_pipedec:]
                    appended_tree_mask = tree_mask[:, :, -run_config.init_topk_pipedec:, :]
                # print(f"draft_tokens: {self.tokenizer.decode(draft_tokens[0])}")
                # print(f'Stage {config.stage} {i}th: appended_tree_mask: {appended_tree_mask.shape}, {appended_tree_mask.dtype}, {appended_tree_mask.element_size()}')
                # if log:
                #     print(f"Stage {config.stage}: draft init forward")
                #     print(f'draft_tokens: {draft_tokens}')
                last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
                # print(f'Stage {config.stage} {i}th: send to next stage...')
                comm.send_appended(appended_input, appended_tree_pos_ids, appended_tree_mask.contiguous())
                # print(f'Stage {config.stage} {i}th: send done!!!')
                lens_split.append(draft_tokens.size(-1) - sum(lens_split))

            lens_split = torch.tensor(lens_split, dtype=torch.long)
            # print(f'Stage {config.stage}: lens_split: {lens_split}')
            ri_cum_depths = []
            path_depth = (retrieve_indices != -1).sum(dim=-1)
            for i in range(config.total_stage):
                cur_depth = torch.full((retrieve_indices.size(0),), i+1, dtype=torch.long)
                subseq_ri_cum_depth = torch.min(cur_depth, path_depth)
                ri_cum_depths.append(subseq_ri_cum_depth)
            subseq_ri_cum_depths = torch.stack(ri_cum_depths, dim=0)
            return lens_split, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, subseq_ri_cum_depths, last_ea_state, last_ea_tree
        
        # following stages 1-4
        for i in range(config.total_stage - config.stage):
            # print(f'Stage {config.stage} {i}th: recv from last stage...')
            with prof.time_context(f"Stage {config.stage} {i}th: recv from last stage", cpu=True) if prof is not None else nullcontext():
                appended_input, subseq_pos_ids, tree_mask = comm.recv_appended(device)
            # print(f'Stage {config.stage} {i}th: recv done!!!')
            # set the tree mask for the current stage
            self.stage_base_model.model.tree_mask = tree_mask

            # print(f'Stage {config.stage} {i}th: draft init forward...')
            with prof.time_context(f"Stage {config.stage}: draft init forward", cpu=False) if prof is not None else nullcontext():
                if config.is_first_stage:
                    # print(f'Stage {config.stage} {i}th: appended_input: {appended_input.shape}, subseq_pos_ids: {subseq_pos_ids}')
                    outputs, sub_hidden_state = self(
                        input_ids=appended_input,
                        past_key_values=stage_past_key_values,
                        position_ids=subseq_pos_ids,
                    )
                else:
                    outputs, sub_hidden_state = self(
                        inputs_embeds=appended_input,
                        past_key_values=stage_past_key_values,
                        position_ids=subseq_pos_ids,
                    )

            if config.is_last_stage:
                comm.sendto(sub_hidden_state, config.next_rank)
            else:
                # print(f'Stage {config.stage}: {type(sub_hidden_state)}')
                comm.send_appended(sub_hidden_state, subseq_pos_ids, tree_mask)
    
    @torch.no_grad()
    def stage_generate(
        self,
        input_ids=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
        pipeline_type="naive",
        profiler=None,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
        # [pipeline method]
        if pipeline_type == "serial":
            pipeline_forward = self._serial_pipeline
        elif pipeline_type == "naive":
            pipeline_forward = self._naive_pipeline
        elif pipeline_type == "pruned":
            pipeline_forward = self._pruned_pipeline
        elif pipeline_type == "continuous":
            pipeline_forward = self._continuous_pipeline
        elif pipeline_type == "pipedec":
            pipeline_forward = self._run_pipedec
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        # [update]: draft stage and last stage need logits_processor
        # if temperature > 1e-5 and (self.is_draft_stage or self.config.is_last_stage):
        if temperature > 1e-5:  # 先让全部stage都有，怕出bug
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        
        # initialize the past key and value states
        if self.is_draft_stage:
            self.ea_layer.reset_kv()
        else:
            self.stage_base_model.model.tree_mask = None
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
                ) = initialize_past_key_values(self.stage_base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data
        # to determine the stop of the pipeline
        should_stop = torch.tensor(0, dtype=torch.int32)  # for break the outer loop
        
        config = self.config
        device = self.stage_base_model.device
        comm = self.comm
        reset_tree_mode(self)
        
        # [update]: 不用padding了，直接用F.pad在后面补-1
        # if config.is_first_stage or config.is_last_stage:
        #     padding = torch.zeros(1, 1, dtype=torch.long) - 1  # padding -1 to the draft token sequence

        # [update] prefill: draft stage recv hidden_state and return orig
        if self.is_draft_stage:
            # [update] get input_ids from the first stage
            input_ids = input_ids.clone()
            input_len = input_ids.shape[1]
            # orig, hidden_state = chunked_prefill(self, input_ids=input_ids, prof=profiler)
            orig, hidden_state = pipeline_prefill(self, input_ids=input_ids, prof=profiler)
            
            new_token = 0
            token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
            global skip_count 
            skip_count = 0

        else:
            # chunked_prefill(self, stage_past_key_values=past_key_values, prof=profiler)
            pipeline_prefill(self, stage_past_key_values=past_key_values, prof=profiler)
        # dist.barrier()
        should_stop = torch.tensor(0, dtype=torch.int32)

        turns_cnt = 0
        # new_sampled_token = -1
        # no kv_cache for the draft stage
        kv_cache=(past_key_values, past_key_values_data, current_length_data) if not self.is_draft_stage else None

        # outer loop
        for idx_spec in range(max_length):
            if config.is_draft_stage:
                outputs = pipeline_forward(
                    logits_processor=logits_processor,
                    input_ids=input_ids,
                    token=token,
                    hidden_state=hidden_state,
                    new_token=new_token,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    log=log,
                    prof=profiler
                )
            elif config.is_first_stage:
                outputs = pipeline_forward(
                    kv_cache=kv_cache,
                    logits_processor=logits_processor,
                    # input_ids=input_ids,
                    # token=token,
                    # hidden_state=hidden_state,
                    # new_token=new_token,
                    log=log,
                    prof=profiler
                )
            else:
                outputs = pipeline_forward(kv_cache, logits_processor, prof=profiler)

            if self.is_draft_stage:
                input_ids, hidden_state, token, accept_length, turns = outputs
                # assert accept_length == len(input_ids[0, input_len:]) - new_token, f'accept_length: {accept_length} != len(input_ids[0, input_len:]) - new_token: {len(input_ids[0, input_len:]) - new_token}'
                new_token += accept_length
                if log:
                    # print(f'{idx_spec}th round, accept_length: {accept_length} in {turns} turns')
                    turns_cnt += turns

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
        if self.is_draft_stage:
            if not log:
                return input_ids
            else:
                # print(f'skip_count: {skip_count}')
                return input_ids, new_token, idx_spec, turns_cnt
            
    def _serial_pipeline(
        self,
        kv_cache=None,
        logits_processor=None,
        input_ids=None,
        token=None,
        hidden_state=None,
        new_token=None,
        max_new_tokens=None,
        max_length=None,
        log=False,
        prof=None
    ):
        config = self.config
        device = self.stage_base_model.device
        comm = self.comm
        if self.is_draft_stage:
            input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.topK_genrate(
                hidden_state,
                input_ids_ea,
                self.stage_base_model.lm_head,
                logits_processor,
            )
            tree_position_ids = tree_position_ids + input_ids.size(-1)
            # send to next stage
            comm.send_appended(draft_tokens, tree_position_ids, tree_mask)
            hidden_state_verified = comm.recvfrom(self.config.last_rank, device)
            logits = self.stage_base_model.lm_head(hidden_state_verified)
            # if self.is_draft_stage:
            # logits, hidden_state = outputs
            logits = logits[0, retrieve_indices]

            # padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(self.stage_base_model.device)
            # draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            draft_tokens = F.pad(draft_tokens, (0, 1), value=-1)

            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(f'accept_length: {accept_length}')
            accept_length += 1

            # prev_input_len = input_ids.size(-1)
            # select_indices = retrieve_indices[best_candidate, :accept_length] + prev_input_len
            # for i in range(1, self.config.world_size):
            #     comm.sendto(select_indices, i)  # 0 must be received, therefore select_indices[0] is the prev_input_len
            # input_ids = torch.cat((input_ids, draft_tokens[0, select_indices]), dim=1)
            # new_token += accept_length

            # token = gen_token(prob=sample_p, logits_processor=logits_processor)
            # return accept_length, self.config.total_stage*2-1
            
        else:
            past_key_values, past_key_values_data, current_length_data = kv_cache
            input_tensor, tree_position_ids, tree_mask = comm.recv_appended(device)
            self.stage_base_model.model.tree_mask = tree_mask
            if self.is_first_stage:
                outputs, hidden_state = self(
                    input_ids=input_tensor,
                    past_key_values=past_key_values,
                    position_ids=tree_position_ids,
                )
            else:
                outputs, hidden_state = self(
                    inputs_embeds=input_tensor,
                    past_key_values=past_key_values,
                    position_ids=tree_position_ids,
                )
            if self.is_last_stage:
                comm.sendto(hidden_state, config.next_rank)
            else:
                comm.send_appended(hidden_state, tree_position_ids, tree_mask)
        
        # [update_inference_inputs]
        if self.is_draft_stage:
            candidates = draft_tokens[0, retrieve_indices].to(input_ids.device)
            update_inputs_params = (
                self,
                None, # past_key_values_data
                None, # current_length_data
                logits_processor,
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                # new_token,
                hidden_state_verified,
                sample_p,
            )
        else:
            update_inputs_params = (self, past_key_values_data, current_length_data)

        outputs = update_stage_inference_inputs(*update_inputs_params)
        if self.is_draft_stage:
            return *outputs, accept_length, self.config.total_stage


    # inner loop
    def _naive_pipeline(
        self,
        kv_cache=None,
        logits_processor=None,
        input_ids=None,
        token=None,
        hidden_state=None,
        new_token=None,
        max_new_tokens=None,
        max_length=None,
        log=False,
        prof=None
    ):  
        if self.is_draft_stage:
            input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.topK_genrate(
                hidden_state,
                input_ids_ea,
                self.stage_base_model.lm_head,
                logits_processor,
                total_tokens=run_config.init_total_token,
                depth=run_config.init_depth,
                top_k=run_config.init_topk,
                return_last=False,
                sort_score=False,
                # prof=prof,
            )
            seqs_split, lens_split = split_sequence_close_equal_len(
                draft_tokens,
                run_config.num_stage
            )
            tree_decoding_params = (
                self, None, seqs_split, lens_split, tree_position_ids, tree_mask, input_ids, prof
            )
        else:
            past_key_values, past_key_values_data, current_length_data = kv_cache
            tree_decoding_params = (self, past_key_values, None, None, None, None, None, prof)

        outputs = stage_tree_decoding(*tree_decoding_params)

        if self.is_draft_stage:
            logits, hidden_state = outputs
            logits = logits[0, retrieve_indices]

            # padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(self.stage_base_model.device)
            # draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            draft_tokens = F.pad(draft_tokens, (0, 1), value=-1)

            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accept_length += 1
            
        # [update_inference_inputs]
        if self.is_draft_stage:
            candidates = draft_tokens[0, retrieve_indices].to(input_ids.device)
            update_inputs_params = (
                self,
                None, # past_key_values_data
                None, # current_length_data
                logits_processor,
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                # new_token,
                hidden_state,
                sample_p,
            )
        else:
            update_inputs_params = (self, past_key_values_data, current_length_data)

        outputs = update_stage_inference_inputs(*update_inputs_params)
        if self.is_draft_stage:
            return *outputs, accept_length, self.config.total_stage*2-1

    def _pruned_pipeline(
        self,
        kv_cache=None,
        logits_processor=None,
        input_ids=None,
        token=None,
        hidden_state=None,
        new_token=None,
        max_new_tokens=None,
        max_length=None,
        log=False,
        prof=None
    ):
        config = self.config
        comm = self.comm
        device = self.stage_base_model.device

        if config.is_draft_stage:
            input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            input_len = input_ids.size(-1)

            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.topK_genrate(
                    hidden_state,
                    input_ids_ea,
                    self.stage_base_model.lm_head,
                    logits_processor,
                    total_tokens=run_config.init_total_token,
                    depth=run_config.init_depth,
                    top_k=run_config.init_topk,
                    return_last=False,
                    sort_score=run_config.draft_gen_sort_score,
                    # prof=prof,
            )
            # todo: 好像没必要在这里移到gpu？因为要传给其他stages
            # update tree_position_ids
            tree_position_ids = tree_position_ids + input_ids.size(-1)
            
            # TODO:这个函数还需要改 最好可以输出一个waiting_draft, 因为最后剩下可以较多
            draft_tokens_split, lens_split, subseq_ri_cum_depths = token_tree_partition(
                draft_tokens, retrieve_indices, run_config.num_stage, run_config.init_subseq_token
            )
            # 仍保留先分成四段
            fill_pipeline_stages(
                self,
                lens_split=lens_split,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_pos_ids=tree_position_ids,
                subseq_ri_cum_depths=subseq_ri_cum_depths,
                prof=prof
            )
            accept_hidden_states = []
            global_accept_len = input_ids.size(-1)
            accept_length_this_round = 0
            lens_split = lens_split
        else:
            past_key_values, past_key_values_data, current_length_data = kv_cache
            global_accept_len = current_length_data[0].item()

            outputs = fill_pipeline_stages(self, past_key_values, prof=prof)
            
            # if config.is_last_stage:
            #     sub_hidden_state, lens_split, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, subseq_ri_cum_depths = outputs
            # else:  # middle stages
            #     sub_hidden_state, lens_split, tree_mask, tree_position_ids = outputs
        
        for i in range(run_config.num_stage):
            # if config.is_draft_stage:
            #     print(f'Stage {config.stage} {i}th turn')
            ###################################################
            # Partially: recv from last stage
            ###################################################
            with prof.time_context(f"Stage {config.stage}: part0, recv from last stage", cpu=True) if prof is not None else nullcontext():
                if config.is_draft_stage or config.stage > i + config.total_stage - run_config.num_stage:
                    sub_hidden_state = comm.recvfrom(config.last_rank, device=device)
                    if sub_hidden_state.size(-1) == 1 and sub_hidden_state.item() == -1:
                        hs_len = 0
                        sub_hidden_state = torch.zeros(1, 0, config.hidden_size, dtype=torch.float16)
                    else:
                        hs_len = sub_hidden_state.size(-2)
                        if not config.is_draft_stage:
                            tree_position_ids = comm.recvfrom(config.last_rank, device=device)
                            tree_mask = comm.recvfrom(config.last_rank, device=device)
                            assert sub_hidden_state.size(1) == tree_mask.size(-2) == tree_position_ids.size(-1), f'Stage {config.stage} {i}th turn recv pruning info: sub_hidden_state: {sub_hidden_state.shape}, tree_mask: {tree_mask.shape}, tree_position_ids: {tree_position_ids.shape}'
                else:
                    hs_len = 0
                    sub_hidden_state = torch.zeros(1, 0, config.hidden_size, dtype=torch.float16)
                    tree_mask = None
                    tree_position_ids = None
                    
            ###################################################
            # All: broadcast pruning info
            ###################################################
            with prof.time_context(f"Stage {config.stage}: part1,broadcast pruning info", cpu=False) if prof is not None else nullcontext():  
                skip_pruning = False  
                if config.is_draft_stage:  # last stage正常pruning
                    if hs_len > 0:
                        with prof.time_context(f"Stage {config.stage}: lm_head", cpu=False) if prof is not None else nullcontext():
                            subseq_logits = self.stage_base_model.lm_head(sub_hidden_state)
                        
                        with prof.time_context(f"Stage {config.stage}: get subseq", cpu=False) if prof is not None else nullcontext():
                            sub_draft_tokens = draft_tokens[:, :lens_split[0]]
                            sub_draft_tokens = F.pad(sub_draft_tokens, (0, 1), value=-1)
                            sub_retrieve_indices = get_subtree_retrieve_indices(retrieve_indices, subseq_ri_cum_depths[0])
                            # subseq_ri_cum_depths = subseq_ri_cum_depths[1:]  # remove the first subseq
                            subtree_logits = subseq_logits[0, sub_retrieve_indices]
                            candidates = sub_draft_tokens[0, sub_retrieve_indices].to(input_ids.device)
                        
                        with prof.time_context(f"Stage {config.stage}: evaluate_posterior", cpu=False) if prof is not None else nullcontext():
                            best_candidate, accept_length, sample_p = evaluate_posterior(
                                subtree_logits, candidates, logits_processor
                            )
                        
                        accept_length += 1
                        new_token += accept_length
                        
                        token = gen_token(prob=sample_p, logits_processor=logits_processor)  # device=cuda

                        cur_draft_depth = subseq_ri_cum_depths[0, best_candidate]
                        # if log:
                            # print(f'- {i}th turn, accept_len/local_depth: {accept_length}/{cur_draft_depth}')
                            
                        sub_hidden_state = sub_hidden_state[:, retrieve_indices[best_candidate, :accept_length]]
                        
                        with prof.time_context(f"Stage {config.stage}: last-stage pruning", cpu=True) if prof is not None else nullcontext():
                            left_indices, truncate = cal_pruning_info(draft_tokens, retrieve_indices, best_candidate, accept_length, token, subseq_ri_cum_depths)
                        
                        if not truncate:
                            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                                truncate = True
                            elif new_token > max_new_tokens:
                                truncate = True
                            elif input_ids.shape[1] > max_length:
                                truncate = True
                                
                        if truncate:  # start new speculation round
                            new_sampled_token = token.item()
                            # if log:
                                # print(f'- {i}th turn truncate')
                        else:
                            new_sampled_token = -1

                        pruning_info = torch.cat((torch.tensor((new_sampled_token, accept_length), dtype=torch.long), left_indices), dim=0).contiguous()
                            
                        if not truncate:  # not truncate: async pruning info broadcast
                            # broadcast_pruning_info_task = comm.executor.submit(
                            #     comm.broadcast_send,
                            #     pruning_info
                            # )
                            comm.broadcast_send(pruning_info)
                        else:  # truncate: sync pruning info broadcast
                            comm.broadcast_send(pruning_info)
                            lens_split = tree_mask = tree_position_ids = None
                    else:
                        skip_pruning = True
                        global skip_count 
                        skip_count += 1
                        # broadcast_pruning_info_task = comm.executor.submit(
                        #     comm.broadcast_send,
                        #     torch.tensor([[-1]], dtype=torch.long)
                        # )  
                        comm.broadcast_send(torch.tensor([[-1]], dtype=torch.long))
                else:
                    with prof.time_context(f"Stage {config.stage}: wait broadcast_pruning", cpu=True) if prof is not None else nullcontext():
                        pruning_info = comm.broadcast_recv(0)
                    
                    if (pruning_info.size(-1) == 1 and pruning_info.item() == -1):
                        skip_pruning = True
                    else:
                        new_sampled_token = pruning_info[0].item()
                        accept_length = pruning_info[1].item()
                        left_indices = pruning_info[2:]
                    
                        truncate = new_sampled_token != -1

            #####################################
            # All: pruning, in this part both draft and first stage do the waiting draft pruning, others are token pruning
            #####################################
            with prof.time_context(f"Stage {config.stage}: part2, pruning", cpu=False) if prof is not None else nullcontext():  
                if not skip_pruning:
                    if config.is_draft_stage:
                        accept_length_this_round += accept_length
                        if not truncate:
                            with prof.time_context(f"Stage {config.stage}: draft_stage_pruning", cpu=True) if prof is not None else nullcontext():
                                # [update] update lens_split in the draft stage
                                draft_tokens, tree_mask, tree_position_ids, retrieve_indices, accepted_tokens, subseq_ri_cum_depths, left_indices, lens_split = draft_stage_pruning(
                                        left_indices, accept_length, draft_tokens, tree_mask, tree_position_ids, retrieve_indices, subseq_ri_cum_depths, lens_split
                                )
                            accepted_tokens = accepted_tokens.to(input_ids.device)
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                            
                    else:
                        if truncate:
                            sub_hidden_state = tree_mask = tree_position_ids = None 
                            
                        with prof.time_context(f"Stage {config.stage}: token_pruning", cpu=True) if prof is not None else nullcontext():
                            past_key_values_data, current_length_data, sub_hidden_state, tree_mask, tree_position_ids = token_pruning(
                                past_key_values_data,
                                current_length_data,
                                None,
                                sub_hidden_state,
                                tree_mask,
                                tree_position_ids,
                                left_indices,
                                global_accept_len,
                                accept_length,
                                config.stage
                            )
                        global_accept_len += accept_length
                            
                # start new speculation round
                    if truncate:
                        if config.is_draft_stage:
                            accept_hidden_states.append(sub_hidden_state)
                            token = torch.tensor([[new_sampled_token]], dtype=torch.long, device=input_ids.device)
                            # accepted_tokens = draft_tokens[:, left_indices].to(input_ids.device)
                            accepted_tokens = draft_tokens[:, left_indices[:accept_length]].to(input_ids.device)
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                        break
                else:  
                    accept_length = 0
                    if config.is_draft_stage:  # [update]: drop the first row as it has been processed
                        lens_split = lens_split[1:]
                        subseq_ri_cum_depths = subseq_ri_cum_depths[1:]
            # print(f'Stage {config.stage} {i}th turn pruning')
            # dist.barrier()
            ####################################
            # Partially: draft stage topk, others forward
            #################################### 
            with prof.time_context(f"Stage {config.stage}: part3, forward", cpu=False) if prof is not None else nullcontext():
                if config.is_draft_stage or config.stage > i + config.total_stage - run_config.num_stage:
                    hs_len = sub_hidden_state.size(1) if hs_len > 0 else 0
                
                if config.is_draft_stage:
                    if hs_len > 0:
                        accept_hidden_states.append(sub_hidden_state)
                    # broadcast_pruning_info_task.result()
                    
                elif config.stage > i + config.total_stage - run_config.num_stage:
                    if hs_len > 0:
                        self.stage_base_model.model.tree_mask = tree_mask
                        assert tree_position_ids.size(0) == tree_mask.size(2)==sub_hidden_state.size(1), f'tree_position_ids.size(0) != tree_mask.size(2): {tree_position_ids.size(0)} and {tree_mask.size(2)}'
                        if config.is_first_stage:  # recv input for first stage forward
                            with prof.time_context(f"Stage {config.stage}: forward", cpu=False) if prof is not None else nullcontext():
                                outputs, sub_hidden_state = self(
                                    input_ids=sub_hidden_state,
                                    past_key_values=past_key_values,
                                    position_ids=tree_position_ids
                                )     
                        else:
                            with prof.time_context(f"Stage {config.stage}: forward", cpu=False) if prof is not None else nullcontext():
                                outputs, sub_hidden_state = self(
                                    inputs_embeds=sub_hidden_state,
                                    past_key_values=past_key_values,
                                    position_ids=tree_position_ids
                                )
                        if config.is_last_stage:
                            comm.sendto(sub_hidden_state, config.next_rank)
                        else:
                            comm.sendto(sub_hidden_state, config.next_rank)
                            comm.sendto(tree_position_ids, config.next_rank)
                            comm.sendto(tree_mask, config.next_rank)
                    else:
                        comm.sendto(torch.tensor([[-1]], dtype=torch.long), config.next_rank)
        #####################################
        # truncate: 一个round结束
        #####################################
        if config.is_draft_stage:
            # [update] draft stage不计入
            turns = i + config.total_stage - 1
            hidden_state = torch.cat(accept_hidden_states, dim=-2)
            # return input_ids, hidden_state, token, new_token, accept_length_this_round, turns    
            return input_ids, hidden_state, token, accept_length_this_round, turns    

    
    def _continuous_pipeline(
        self,
        kv_cache=None,
        logits_processor=None,
        input_ids=None,
        token=None,
        hidden_state=None,
        new_token=None,
        max_new_tokens=None,
        max_length=None,
        log=False,
        prof=None
    ):
        config = self.config
        comm = self.comm
        device = self.stage_base_model.device

        if config.is_draft_stage:
            input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            input_len = input_ids.size(-1)

            with prof.time_context(f"Stage {config.stage}: topK_genrate", cpu=False) if prof is not None else nullcontext():
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.topK_genrate(
                        hidden_state,
                        input_ids_ea,
                        self.stage_base_model.lm_head,
                        logits_processor,
                        total_tokens=run_config.init_total_token,
                        depth=run_config.init_depth,
                        top_k=run_config.init_topk,
                        return_last=run_config.none_expand,
                        sort_score=run_config.draft_gen_sort_score,
                        # prof=prof
                )
            if run_config.none_expand:
                last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
                
            tree_position_ids = tree_position_ids + input_ids.size(-1)
            
            draft_tokens_split, lens_split, subseq_ri_cum_depths = token_tree_partition(
                draft_tokens, retrieve_indices, run_config.num_stage, run_config.init_subseq_token
            )
            # 仍保留先分成四段
            fill_pipeline_stages(
                self,
                lens_split=lens_split,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_pos_ids=tree_position_ids,
                subseq_ri_cum_depths=subseq_ri_cum_depths,
                prof=prof
            )
            waiting_draft = 0
            accept_hidden_states = []
            global_accept_len = input_ids.size(-1)
            accept_length_this_round = 0
            lens_split = lens_split  # 跳过最开始?
        else:
            past_key_values, past_key_values_data, current_length_data = kv_cache
            global_accept_len = current_length_data[0].item()

            outputs = fill_pipeline_stages(self, past_key_values, prof=prof)

            # if config.is_last_stage:
            #     sub_hidden_state, lens_split, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, subseq_ri_cum_depths = outputs
            # else:  # middle stages
            #     sub_hidden_state, lens_split, tree_mask, tree_position_ids = outputs
        
        i = -1
        while True:
            i += 1
            # if config.is_draft_stage:
            #     print(f'Stage {config.stage} {i}th turn lens_split: {lens_split}')
            ###################################################
            # recv from last stage
            ###################################################
            with prof.time_context(f"Stage {config.stage}: part0, recv from last stage", cpu=True) if prof is not None else nullcontext():
                sub_hidden_state = comm.recvfrom(config.last_rank, device=device)
                if sub_hidden_state.size(-1) == 1 and sub_hidden_state.item() == -1:
                    hs_len = 0
                    sub_hidden_state = torch.zeros(1, 0, config.hidden_size, dtype=torch.float16)
                else:
                    hs_len = sub_hidden_state.size(-2)
                    if not config.is_draft_stage:
                        tree_position_ids = comm.recvfrom(config.last_rank, device=device)
                        tree_mask = comm.recvfrom(config.last_rank, device=device)
                        assert sub_hidden_state.size(1) == tree_mask.size(-2) == tree_position_ids.size(-1), f'Stage {config.stage} {i}th turn recv pruning info: sub_hidden_state: {sub_hidden_state.shape}, tree_mask: {tree_mask.shape}, tree_position_ids: {tree_position_ids.shape}'
            ###################################################
            # broadcast pruning info
            ###################################################
            # with prof.time_context(f"Stage {config.stage}: part1,broadcast pruning info", cpu=False) if prof is not None else nullcontext():  
            with nullcontext():
                skip_pruning = False
                if config.is_draft_stage:  # last stage正常pruning
                    if hs_len > 0:
                        with prof.time_context(f"Stage {config.stage}: verification and pruning", cpu=False) if prof is not None else nullcontext():
                        # with prof.time_context(f"Stage {config.stage}: lm_head", cpu=False) if prof is not None else nullcontext():
                            subseq_logits = self.stage_base_model.lm_head(sub_hidden_state)
                        
                        # with prof.time_context(f"Stage {config.stage}: get subseq", cpu=False) if prof is not None else nullcontext():
                            sub_draft_tokens = draft_tokens[:, :lens_split[0]]
                            sub_draft_tokens = F.pad(sub_draft_tokens, (0, 1), value=-1)
                            sub_retrieve_indices = get_subtree_retrieve_indices(retrieve_indices, subseq_ri_cum_depths[0])
                            # subseq_ri_cum_depths = subseq_ri_cum_depths[1:]  # remove the first subseq
                            subtree_logits = subseq_logits[0, sub_retrieve_indices]
                            
                            candidates = sub_draft_tokens[0, sub_retrieve_indices]
                        
                        # with prof.time_context(f"Stage {config.stage}: evaluate_posterior", cpu=False) if prof is not None else nullcontext():
                            best_candidate, accept_length, sample_p = evaluate_posterior(
                                subtree_logits, candidates, logits_processor
                            )
                        
                            accept_length += 1
                            new_token += accept_length
                            token = gen_token(prob=sample_p, logits_processor=logits_processor)  # device=cuda

                            cur_draft_depth = subseq_ri_cum_depths[0, best_candidate]
                            # if log:
                                # print(f'- {i}th turn, accept_len/local_depth: {accept_length}/{cur_draft_depth}')
                                
                            sub_hidden_state = sub_hidden_state[:, retrieve_indices[best_candidate, :accept_length]]
                        # with prof.time_context(f"Stage {config.stage}: last-stage pruning", cpu=False) if prof is not None else nullcontext():
                            left_indices, truncate = cal_pruning_info(draft_tokens, retrieve_indices, best_candidate, accept_length, token, subseq_ri_cum_depths)
                        
                        if not truncate:
                            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                                truncate = True
                            elif new_token > max_new_tokens:
                                truncate = True
                            elif input_ids.shape[1] > max_length:
                                truncate = True
                                
                        if truncate:  # start new speculation round
                            new_sampled_token = token.item()
                            # if log:
                                # print(f'- {i}th turn truncate')
                        else:
                            new_sampled_token = -1
                            
                        pruning_info = torch.cat((torch.tensor((new_sampled_token, accept_length), dtype=torch.long), left_indices), dim=0).contiguous()
                            
                        if not truncate:  # not truncate: async pruning info broadcast
                            broadcast_pruning_info_task = comm.executor.submit(
                                comm.broadcast_send,
                                pruning_info
                            )
                            # comm.broadcast_send(pruning_info)
                        else:  # truncate: sync pruning info broadcast
                            comm.broadcast_send(pruning_info)
                            lens_split = tree_mask = tree_position_ids = None
                    
                    else:
                        skip_pruning = True
                        global skip_count 
                        skip_count += 1
                        broadcast_pruning_info_task = comm.executor.submit(
                            comm.broadcast_send,
                            torch.tensor([[-1]], dtype=torch.long)
                        )  
                        # comm.broadcast_send(torch.tensor([[-1]], dtype=torch.long))
                else:
                    with prof.time_context(f"Stage {config.stage}: wait broadcast_pruning", cpu=True) if prof is not None else nullcontext():
                        pruning_info = comm.broadcast_recv(0)
                    
                    if (pruning_info.size(-1) == 1 and pruning_info.item() == -1):
                        skip_pruning = True
                    else:
                        new_sampled_token = pruning_info[0].item()
                        accept_length = pruning_info[1].item()
                        left_indices = pruning_info[2:]
                    
                        truncate = new_sampled_token != -1

            #####################################
            # pruning, in this part both draft and first stage do the waiting draft pruning, others are token pruning
            #####################################
            # with prof.time_context(f"Stage {config.stage}: part2, pruning", cpu=False) if prof is not None else nullcontext():  
            with nullcontext():
                if not skip_pruning:
                    if config.is_draft_stage:
                        accept_length_this_round += accept_length
                        if not truncate:
                            with prof.time_context(f"Stage {config.stage}: draft_stage_pruning", cpu=True) if prof is not None else nullcontext():
                                # [update] update lens_split in the draft stage
                                draft_tokens, tree_mask, tree_position_ids, retrieve_indices, accepted_tokens, subseq_ri_cum_depths, left_indices, lens_split = draft_stage_pruning(
                                        left_indices, accept_length, draft_tokens, tree_mask, tree_position_ids, retrieve_indices, subseq_ri_cum_depths, lens_split
                                )
                            accepted_tokens = accepted_tokens.to(input_ids.device)
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                            waiting_draft = (draft_tokens.size(-1) - torch.sum(lens_split)).item()
                            
                    else:
                        if truncate:
                            sub_hidden_state = tree_mask = tree_position_ids = None 
                        
                        with prof.time_context(f"Stage {config.stage}: token_pruning", cpu=True) if prof is not None else nullcontext():
                            past_key_values_data, current_length_data, sub_hidden_state, tree_mask, tree_position_ids = token_pruning(
                                past_key_values_data,
                                current_length_data,
                                None,
                                sub_hidden_state,
                                tree_mask,
                                tree_position_ids,
                                left_indices,
                                global_accept_len,
                                accept_length,
                                config.stage
                            )
                            global_accept_len += accept_length
                            
                    if truncate:
                        if config.is_draft_stage:
                            accept_hidden_states.append(sub_hidden_state)
                            token = torch.tensor([[new_sampled_token]], dtype=torch.long, device=input_ids.device)
                            # accepted_tokens = draft_tokens[:, left_indices].to(input_ids.device)
                            accepted_tokens = draft_tokens[:, left_indices[:accept_length]].to(input_ids.device)
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                        break
                else:  
                    accept_length = 0
                    if config.is_draft_stage:  # [update]: drop the first row as it has been processed
                        lens_split = lens_split[1:]
                        subseq_ri_cum_depths = subseq_ri_cum_depths[1:]
            ####################################
            # 前向: draft stage topk, others forward
            #################################### 
            # with prof.time_context(f"Stage {config.stage}: part3, forward", cpu=False) if prof is not None else nullcontext():
            with nullcontext():
                hs_len = sub_hidden_state.size(1) if hs_len > 0 else 0
                
                if config.is_draft_stage:
                    cur_draft_depth = (retrieve_indices != -1).sum(dim=1).max()
                    pruned = accept_hidden_states or hs_len
                    
                    if pruned:
                        with prof.time_context(f"Stage {config.stage}: tree_expansion", cpu=False) if prof is not None else nullcontext():
                            # new_token不需要通过hidden_state生成，直接根据剪枝的token生成，新token就是剪枝后新树的根节点
                            new_ea_token = draft_tokens[:, :1].to(input_ids.device)
                            input_ids_ea = torch.cat((input_ids, new_ea_token), dim=-1)
                            
                            if hs_len > 0:
                                accept_hidden_states.append(sub_hidden_state)
                            accepted_hidden_state = torch.cat(accept_hidden_states, dim=-2)
                            accept_hidden_states = []

                            # with prof.time_context(f"Stage {config.stage}: topK_genrate", cpu=False) if prof is not None else nullcontext():
                            draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2, last_ea_state = self.ea_layer.topK_genrate(
                                accepted_hidden_state,
                                input_ids_ea,
                                self.stage_base_model.lm_head,
                                logits_processor,
                                # total_tokens=run_config.expand_total_token - run_config.none_expand_size if lens_split[0] == 0 and run_config.none_expand else run_config.expand_total_token,
                                total_tokens=run_config.expand_total_token,
                                # depth=max(cur_draft_depth + 2, run_config.expand_depth),  # todo: test best tree settings
                                # depth=run_config.expand_depth - run_config.none_expand_depth if lens_split[0] == 0 and run_config.none_expand else run_config.expand_depth,
                                depth=run_config.expand_depth,
                                top_k=run_config.expand_topk,
                                return_last=run_config.none_expand,
                                sort_score=run_config.draft_gen_sort_score,
                                # prof=prof
                            )  # get a little more appended tokens
                            
                            if run_config.none_expand:
                                last_ea_tree = (draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2)
                            assert retrieve_indices2.size(-1) > max(tree_position_ids2), f'retrieve_indices2.size(-1) > max(tree_position_ids2): {retrieve_indices2.size(-1)} and {max(tree_position_ids2)}'
                            tree_position_ids2 = tree_position_ids2 + input_ids.size(-1)
                            assert draft_tokens.size(-1) == tree_position_ids.size(0), f'draft_tokens != tree_pos_ids: {draft_tokens.size(-1)} and {tree_position_ids.size(0)}'

                            origin_device = draft_tokens.device
                            # with prof.time_context(f"Stage {config.stage}: topk merge_two_tree", cpu=True) if prof is not None else nullcontext():
                            # [update] operate on CPU
                            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, lens_split, subseq_ri_cum_depths = merge_two_tree(
                                (draft_tokens.cpu(), retrieve_indices.cpu(), tree_mask.cpu(), tree_position_ids.cpu()),
                                (draft_tokens2.cpu(), retrieve_indices2.cpu(), tree_mask2.cpu(), tree_position_ids2.cpu()),
                                lens_split,
                                subseq_ri_cum_depths
                            )
                            draft_tokens = draft_tokens.to(origin_device)
                            tree_mask = tree_mask.to(origin_device)
                            tree_position_ids = tree_position_ids.to(origin_device)
                                
                        waiting_draft = lens_split[-1].item()

                        appended_draft_len = min(waiting_draft, run_config.expand_subseq_token) if run_config.expand_subseq_token != -1 else waiting_draft
                        lens_split[-1] = appended_draft_len

                    else:
                        if run_config.none_expand and (last_ea_state is not None):
                            with prof.time_context(f"Stage {config.stage}: tree_expand_last", cpu=False) if prof is not None else nullcontext():
                                # with prof.time_context(f"Stage {config.stage}: expand_last", cpu=False) if prof is not None else nullcontext():
                                draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2, last_ea_state = self.ea_layer.expand_last(
                                    last_ea_tree,
                                    last_ea_state,
                                    self.stage_base_model.lm_head,
                                    logits_processor,
                                    device,
                                    expand_depth=run_config.none_expand_depth,
                                    expand_size=run_config.none_expand_size,
                                    return_last=run_config.none_expand,
                                    # prof=prof
                                )
                                last_ea_tree = (draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2)
                                tree_position_ids2 = tree_position_ids2 + input_ids.size(-1)
                                
                                origin_device = draft_tokens.device
                                # with prof.time_context(f"Stage {config.stage}: expand_last merge_two_tree", cpu=True) if prof is not None else nullcontext():
                                # [update] operate on CPU
                                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, lens_split, subseq_ri_cum_depths = merge_two_tree(
                                    (draft_tokens.cpu(), retrieve_indices.cpu(), tree_mask.cpu(), tree_position_ids.cpu()),
                                    (draft_tokens2.cpu(), retrieve_indices2.cpu(), tree_mask2.cpu(), tree_position_ids2.cpu()),
                                    lens_split,
                                    subseq_ri_cum_depths
                                )
                                draft_tokens = draft_tokens.to(origin_device)
                                tree_mask = tree_mask.to(origin_device)
                                tree_position_ids = tree_position_ids.to(origin_device)
                                    
                                waiting_draft = lens_split[-1].item()

                                # print(f'Stage {config.stage}: expand_last waiting_draft: {waiting_draft}')

                                appended_draft_len = min(waiting_draft, run_config.expand_subseq_token)  if run_config.expand_subseq_token != -1 else waiting_draft
                                lens_split[-1] = appended_draft_len
                        else:
                            appended_draft_len = min(waiting_draft, run_config.expand_subseq_token) if run_config.expand_subseq_token != -1 else waiting_draft
                            lens_split = torch.cat((lens_split, torch.tensor([appended_draft_len], dtype=torch.long)))
                        
                    waiting_draft -= appended_draft_len
                    cur_subseq_ri_cum_depth = subseq_ri_cum_depths[-1].clone()
                    
                    if appended_draft_len > 0:
                        existing_draft_len = torch.sum(lens_split[:-1])  # log: existing_draft_len: 0; appended_draft_len: 10; input_draft_end_idx: 10???
                        input_draft_end_idx = existing_draft_len + appended_draft_len

                        retrieve_indices_filled = torch.cat((retrieve_indices, torch.full((retrieve_indices.size(0), 1), -1, dtype=torch.long)), dim=1)

                        for j in range(existing_draft_len, input_draft_end_idx):
                            row_indices = torch.arange(retrieve_indices.size(0), dtype=torch.long)
                            cum_ri_leaves = retrieve_indices_filled[row_indices, cur_subseq_ri_cum_depth]
                            cur_subseq_ri_cum_depth[cum_ri_leaves == j] += 1
            
                        appended_draft_tokens = draft_tokens[..., existing_draft_len:input_draft_end_idx].contiguous()
                        appended_tree_position_ids = tree_position_ids[existing_draft_len:input_draft_end_idx].contiguous()
                        appended_tree_mask = tree_mask[..., existing_draft_len:input_draft_end_idx, :input_draft_end_idx].contiguous()
                        comm.sendto(appended_draft_tokens, config.next_rank)
                        comm.sendto(appended_tree_position_ids, config.next_rank)
                        comm.sendto(appended_tree_mask, config.next_rank)
                    else:
                        comm.sendto(torch.tensor([[-1]], dtype=torch.long), config.next_rank)
                        
                    subseq_ri_cum_depths = torch.cat((subseq_ri_cum_depths, cur_subseq_ri_cum_depth.unsqueeze(0)), dim=0)
                    # broadcast_pruning_info_task.result()
                    
                else:
                    if hs_len > 0:
                        self.stage_base_model.model.tree_mask = tree_mask
                        assert tree_position_ids.size(0) == tree_mask.size(2)==sub_hidden_state.size(1), f'tree_position_ids.size(0) != tree_mask.size(2): {tree_position_ids.size(0)} and {tree_mask.size(2)}'
                        with prof.time_context(f"Stage {config.stage}: forward", cpu=False) if prof is not None else nullcontext():
                            if config.is_first_stage:  # recv input for first stage forward
                                outputs, sub_hidden_state = self(
                                    input_ids=sub_hidden_state,
                                    past_key_values=past_key_values,
                                    position_ids=tree_position_ids
                                )     
                            else:
                                outputs, sub_hidden_state = self(
                                    inputs_embeds=sub_hidden_state,
                                    past_key_values=past_key_values,
                                    position_ids=tree_position_ids
                                )
                        if config.is_last_stage:
                            comm.sendto(sub_hidden_state, config.next_rank)
                        else:
                            comm.sendto(sub_hidden_state, config.next_rank)
                            comm.sendto(tree_position_ids, config.next_rank)
                            comm.sendto(tree_mask, config.next_rank)
                    else:
                        comm.sendto(torch.tensor([[-1]], dtype=torch.long), config.next_rank)
        #####################################
        # truncate: 一个round结束
        #####################################
        if config.is_draft_stage:
            # [update] draft stage不计入
            turns = i + config.total_stage - 1
            hidden_state = torch.cat(accept_hidden_states, dim=-2)
            # return input_ids, hidden_state, token, new_token, accept_length_this_round, turns    
            return input_ids, hidden_state, token, accept_length_this_round, turns
        
    def _run_pipedec(
        self,
        kv_cache=None,
        logits_processor=None,
        input_ids=None,
        token=None,
        hidden_state=None,
        new_token=None,
        max_new_tokens=None,
        max_length=None,
        log=False,
        prof=None
    ):
        config = self.config
        comm = self.comm
        device = self.stage_base_model.device

        # [draft init: fill pipeline stages]
        if config.is_draft_stage:
            # input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # if log:
            #     print
            draft_init_params = (None, input_ids, token.to(input_ids.device), hidden_state, logits_processor)
            global_accept_len = input_ids.size(-1)
            input_len = input_ids.size(-1)
        else:
            past_key_values, past_key_values_data, current_length_data = kv_cache
            draft_init_params = (past_key_values,)
            global_accept_len = current_length_data[0].item()

        outputs = self.draft_init_pipedec(*draft_init_params)
        # print(f'Stage {config.stage}: draft init done!!!')


        if config.is_draft_stage:
            lens_split, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, subseq_ri_cum_depths, last_ea_state, last_ea_tree = outputs
            accept_hidden_states = []
            accept_length_this_round = 0
            accept_tokens = None
        # [end]
        dist.barrier()

        i = -1
        while True:
            i += 1
            ####################################
            # recv from last stage
            ####################################
            # if log:
                # print(f'Stage {config.stage} {i}th: recv from last stage...')
            sub_hidden_state = comm.recvfrom(config.last_rank, device=device)
            if sub_hidden_state.size(-1) == 1 and sub_hidden_state.item() == -1:
                hs_len = 0
                sub_hidden_state = torch.zeros(1, 0, config.hidden_size, dtype=torch.float16)
            else:
                hs_len = sub_hidden_state.size(-2)
                if not config.is_draft_stage:
                    tree_position_ids = comm.recvfrom(config.last_rank, device=device)
                    tree_mask = comm.recvfrom(config.last_rank, device=device)
                    assert sub_hidden_state.size(1) == tree_mask.size(-2) == tree_position_ids.size(-1), f'Stage {config.stage} {i}th turn recv pruning info: sub_hidden_state: {sub_hidden_state.shape}, tree_mask: {tree_mask.shape}, tree_position_ids: {tree_position_ids.shape}'
            # if log:
                # print(f'Stage {config.stage} {i}th: recv done!!!')
            
            
            ####################################
            # broadcast pruning info
            ####################################
            skip_pruning = False
            if config.is_draft_stage:  # last stage正常pruning
                if hs_len > 0:
                    # print(f'draft_tokens: {self.tokenizer.decode(draft_tokens[0])}')
                    with prof.time_context(f"Stage {config.stage}: verification and pruning", cpu=False) if prof is not None else nullcontext():
                    # with prof.time_context(f"Stage {config.stage}: lm_head", cpu=False) if prof is not None else nullcontext():
                        subseq_logits = self.stage_base_model.lm_head(sub_hidden_state)
                    
                    # with prof.time_context(f"Stage {config.stage}: get subseq", cpu=False) if prof is not None else nullcontext():
                        sub_draft_tokens = draft_tokens[:, :lens_split[0]]
                        # print(f'sub_draft_tokens: {sub_draft_tokens.shape}')
                        sub_draft_tokens = F.pad(sub_draft_tokens, (0, 1), value=-1)
                        # print(f'subseq_ri_cum_depth: {subseq_ri_cum_depths[0]}')
                        sub_retrieve_indices = get_subtree_retrieve_indices(retrieve_indices, subseq_ri_cum_depths[0])
                        # print(f'sub_retrieve_indices: {sub_retrieve_indices.shape}')
                        # subseq_ri_cum_depths = subseq_ri_cum_depths[1:]  # remove the first subseq
                        subtree_logits = subseq_logits[0, sub_retrieve_indices]
                        # print(f'subtree_logits: {subtree_logits.shape}')
                        
                        candidates = sub_draft_tokens[0, sub_retrieve_indices]
                    
                    # with prof.time_context(f"Stage {config.stage}: evaluate_posterior", cpu=False) if prof is not None else nullcontext():
                        best_candidate, accept_length, sample_p = evaluate_posterior(
                            subtree_logits, candidates, logits_processor
                        )
                        
                        # best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
                        accept_length += 1
                        new_token += accept_length
                        # sample_p = subseq_logits[0, 0]
                        
                        token = gen_token(prob=sample_p, logits_processor=logits_processor)  # device=cuda
                        # print(f'sample Token is ({self.tokenizer.decode(token.item())})')

                        cur_draft_depth = subseq_ri_cum_depths[0, best_candidate]
                        # if log:
                            # print(f'- {i}th turn, accept_len/local_depth: {accept_length}/{cur_draft_depth}')

                        # sub_hidden_state = sub_hidden_state[:, retrieve_indices[best_candidate, :accept_length]]
                        # assert sub_draft_tokens.size(1) == 1, f'sub_draft_tokens.shape: {sub_draft_tokens.shape}'
                        assert sub_draft_tokens[0, 0] == sub_draft_tokens[0, retrieve_indices[best_candidate, 0]], f'sub_draft_tokens[0, 0] != sub_draft_tokens[0, retrieve_indices[best_candidate, 0]]: {sub_draft_tokens[0, 0]} and {sub_draft_tokens[0, retrieve_indices[best_candidate, 0]]}'
                    # with prof.time_context(f"Stage {config.stage}: last-stage pruning", cpu=False) if prof is not None else nullcontext():
                        left_indices, truncate = cal_pruning_info(draft_tokens, retrieve_indices, best_candidate, accept_length, token, subseq_ri_cum_depths)
                        # print(f'left_indices.shape: {left_indices.shape}')
                        # print(f'left_indices: {self.tokenizer.decode(draft_tokens[left_indices[0]])}, truncate: {truncate}')
                    
                    if not truncate:
                        if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                            truncate = True
                        elif new_token > max_new_tokens:
                            truncate = True
                        elif input_ids.shape[1] > max_length:
                            truncate = True
                                
                    if truncate:  # start new speculation round
                        new_sampled_token = token.item()
                        # if log:
                        #     print(f'- {i}th turn truncate')
                    else:
                        new_sampled_token = -1
                        
                    pruning_info = torch.cat((torch.tensor((new_sampled_token, accept_length), dtype=torch.long), left_indices), dim=0).contiguous()
                    
                    # print(f'Stage {config.stage} {i}th: send pruning_info: {pruning_info}')
                    if not truncate:  # not truncate: async pruning info broadcast
                        # broadcast_pruning_info_task = comm.executor.submit(
                        #     comm.broadcast_send,
                        #     pruning_info
                        # )
                        comm.broadcast_send(pruning_info)
                    else:  # truncate: sync pruning info broadcast
                        comm.broadcast_send(pruning_info)
                        lens_split = tree_mask = tree_position_ids = None
                
                else:
                    # print(f'Stage {config.stage} {i}th: send pruning_info with hs_len = 0')
                    skip_pruning = True
                    left_indices = None
                    global skip_count 
                    skip_count += 1
                    # broadcast_pruning_info_task = comm.executor.submit(
                    #     comm.broadcast_send,
                    #     torch.tensor([[-1]], dtype=torch.long)
                    # )  
                    comm.broadcast_send(torch.tensor([[-1]], dtype=torch.long))
            else:
                # print(f'Stage {config.stage} {i}th: wait broadcast_pruning...')
                with prof.time_context(f"Stage {config.stage}: wait broadcast_pruning", cpu=True) if prof is not None else nullcontext():
                    pruning_info = comm.broadcast_recv(0)
                # print(f'Stage {config.stage} {i}th: wait broadcast_pruning done!!!')
                
                if (pruning_info.size(-1) == 1 and pruning_info.item() == -1):
                    skip_pruning = True
                else:
                    new_sampled_token = pruning_info[0].item()
                    accept_length = pruning_info[1].item()
                    left_indices = pruning_info[2:]
                
                    truncate = new_sampled_token != -1

            ####################################
            # pruning
            ####################################
            # print(f'Stage {config.stage} {i}th: pruning...')
            if not skip_pruning:
                if config.is_draft_stage:
                    accept_length_this_round += accept_length
                    if not truncate:
                        with prof.time_context(f"Stage {config.stage}: draft_stage_pruning", cpu=True) if prof is not None else nullcontext():
                            # [update] update lens_split in the draft stage
                            draft_tokens, tree_mask, tree_position_ids, retrieve_indices, accepted_tokens, subseq_ri_cum_depths, left_indices, lens_split = draft_stage_pruning(
                                    left_indices, accept_length, draft_tokens, tree_mask, tree_position_ids, retrieve_indices, subseq_ri_cum_depths, lens_split
                            )
                            # last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
                            # print(f'Stage {config.stage} {i}th: draft_stage_pruning done!!!: tree_position_ids: {tree_position_ids.shape}, tree_mask: {tree_mask.shape}, retrieve_indices: {retrieve_indices.shape}, accepted_tokens: {accepted_tokens.shape}')
                        accepted_tokens = accepted_tokens.to(input_ids.device)
                        input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                        # print(f'Stage {config.stage} {i}th: input_ids: {self.tokenizer.decode(input_ids[0])}')
                        if accept_tokens is None:
                            accept_tokens = accepted_tokens
                        else:
                            accept_tokens = torch.cat((accept_tokens, accepted_tokens), dim=-1)
                        waiting_draft = (draft_tokens.size(-1) - torch.sum(lens_split)).item()
                else:
                    if truncate:
                        sub_hidden_state = tree_mask = tree_position_ids = None 
                    
                    past_key_values_data, current_length_data, sub_hidden_state, tree_mask, tree_position_ids = token_pruning(
                        past_key_values_data,
                        current_length_data,
                        None,
                        sub_hidden_state,
                        tree_mask,
                        tree_position_ids,
                        left_indices,
                        global_accept_len,
                        accept_length,
                        config.stage
                    )
                    global_accept_len += accept_length
                
                if truncate:
                    if config.is_draft_stage:
                        accept_hidden_states.append(sub_hidden_state)
                        token = torch.tensor([[new_sampled_token]], dtype=torch.long, device=input_ids.device)
                        # accept_tokens = draft_tokens[:, left_indices].to(input_ids.device)
                        accept_tokens = draft_tokens[:, left_indices[:accept_length]].to(input_ids.device)
                        input_ids = torch.cat((input_ids, accept_tokens), dim=-1)
                    # print(f'Stage {config.stage} {i}th: truncate, break!!!')
                    break
            
            else:
                accept_length = 0
                if config.is_draft_stage:
                    lens_split = lens_split[1:]
                    subseq_ri_cum_depths = subseq_ri_cum_depths[1:]

            ####################################
            # draft stage topk, others forward
            ####################################
            hs_len = sub_hidden_state.size(1) if hs_len > 0 else 0
            
            if config.is_draft_stage:
                # print(f'Stage {config.stage} {i}th: subseq_ri_cum_depths after pruning: {subseq_ri_cum_depths}')
                # print(f'lens_split: {lens_split}')

                
                cur_draft_depth = (retrieve_indices != -1).sum(dim=1).max()
                pruned = accept_hidden_states or hs_len

                if pruned:
                    new_ea_token = draft_tokens[:, :1].to(input_ids.device)
                    input_ids_ea = torch.cat((input_ids, new_ea_token), dim=-1)
                    
                    if hs_len > 0:
                        accept_hidden_states.append(sub_hidden_state)
                    # pipedec延申不更新上下文
                    # accepted_hidden_state = torch.cat(accept_hidden_states, dim=-2)
                    # accept_hidden_states = []

                    # 应该不需要merge了
                    last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.expand_pipedec(
                        None,
                        input_ids,
                        self.stage_base_model.lm_head,
                        logits_processor,
                        top_k=run_config.init_topk_pipedec,
                        log=log,
                        first_expand=False,
                        last_state=last_ea_state,
                        tree=last_ea_tree,
                        accept_tokens=accept_tokens,
                        left_indices=left_indices
                    )
                    last_ea_tree = (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
                    # with open('draft_tokens.txt', 'a') as f:
                    #     f.write(f'{self.tokenizer.decode(draft_tokens[0])}\n')

                    # n_leaves = retrieve_indices.size(0)
                    # subseq_ri_cum_depths = []
                    # cum_seq_lens = np.cumsum(lens_split[:-1].numpy(), axis=0)
                    # bottom = np.full((n_leaves, 1), -1, dtype=np.int64)
                    # retrieve_indices_filled = np.concatenate((retrieve_indices.numpy(), bottom), axis=1)  # add -1 to bottom to prevent overflow

                    # ri_depth_cum = np.zeros(n_leaves, dtype=np.int64)
                    # for i, cum_seq_len in enumerate(cum_seq_lens):
                    #     for j in range(0 if i == 0 else cum_seq_lens[i - 1], cum_seq_len):
                    #         row_indices = np.arange(n_leaves, dtype=np.int64)
                    #         cum_ri_leaves = retrieve_indices_filled[row_indices, ri_depth_cum]
                    #         ri_depth_cum[cum_ri_leaves == j] += 1
                    #     # update: 只计算到在pipeline里的draft token tree部分，即将输入的最新一段单独算
                    #     subseq_ri_cum_depths.append(ri_depth_cum.copy())
                    # subseq_ri_cum_depths = np.stack(subseq_ri_cum_depths, axis=0)

                    # subseq_ri_cum_depth = (retrieve_indices != -1).sum(dim=1)[None]
                    # subseq_ri_cum_depths = np.concatenate((subseq_ri_cum_depths, subseq_ri_cum_depth), axis=0)

                    # subseq_ri_cum_depths = torch.from_numpy(subseq_ri_cum_depths)
                    subseq_ri_cum_depths = get_subseq_ri_cum_depths(retrieve_indices, lens_split)
                    # print(f'Stage {config.stage} {i}th: subseq_ri_cum_depths after expand: {subseq_ri_cum_depths}')

                    lens_split = torch.cat((lens_split, torch.tensor([run_config.init_topk_pipedec], dtype=torch.long)), dim=0)
                    waiting_draft = lens_split[-1].item()
                    # print(f'lens_split after expand: {lens_split}')

                    appended_draft_len = waiting_draft
                    lens_split[-1] = appended_draft_len

                    appended_draft_tokens = draft_tokens[:, -appended_draft_len:]
                    appended_tree_position_ids = tree_position_ids[-appended_draft_len:]
                    appended_tree_mask = tree_mask[:,:, -appended_draft_len:, :]
                    comm.sendto(appended_draft_tokens, config.next_rank)
                    comm.sendto(appended_tree_position_ids, config.next_rank)
                    comm.sendto(appended_tree_mask, config.next_rank)

                    # broadcast_pruning_info_task.result()

                    # path_depths = (retrieve_indices != -1).sum(dim=1)
                    # cur_depth = torch.full((retrieve_indices.size(0),), i+1, dtype=torch.long)
                    # cur_subseq_ri_cum_depth = torch.min(cur_depth, path_depths).unsqueeze(0)
                    # subseq_ri_cum_depths = torch.cat((subseq_ri_cum_depths, cur_subseq_ri_cum_depth), dim=0)
                
            else:
                if hs_len > 0:
                    self.stage_base_model.model.tree_mask = tree_mask
                    assert tree_position_ids.size(0) == tree_mask.size(2)==sub_hidden_state.size(1), f'tree_position_ids.size(0) != tree_mask.size(2): {tree_position_ids.size(0)} and {tree_mask.size(2)}'

                    if config.is_first_stage:
                        outputs, sub_hidden_state = self(
                            input_ids=sub_hidden_state,
                            past_key_values=past_key_values,
                            position_ids=tree_position_ids
                        )
                    else:
                        outputs, sub_hidden_state = self(
                            inputs_embeds=sub_hidden_state,
                            past_key_values=past_key_values,
                            position_ids=tree_position_ids
                        )
                    if config.is_last_stage:
                        comm.sendto(sub_hidden_state, config.next_rank)
                    else:
                        comm.sendto(sub_hidden_state, config.next_rank)
                        comm.sendto(tree_position_ids, config.next_rank)
                        comm.sendto(tree_mask, config.next_rank)
                else:
                    comm.sendto(torch.tensor([[-1]], dtype=torch.long), config.next_rank)
        #####################################
        # truncate: 一个round结束
        #####################################
        if config.is_draft_stage:
            # [update] draft stage不计入
            turns = i + config.total_stage - 1
            hidden_state = torch.cat(accept_hidden_states, dim=-2)
            # return input_ids, hidden_state, token, new_token, accept_length_this_round, turns    
            return input_ids, hidden_state, token, accept_length_this_round, turns 

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

        # [MODIFIED]: only first stage and last stage has the ea_layer, embedding, lm_head and input_ids
        if temperature > 1e-5 and (self.is_first_stage or self.is_last_stage):
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # Initialize the past key and value states
        self.stage_base_model.model.tree_mask = None
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
        
        if self.is_first_stage:
            max_length = max_length - self.ea_layer.total_tokens - 10
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoid modifying the input_ids in-place
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

            input_len = input_ids.shape[1]
            reset_tree_mode(self)
            new_token = 0

            # prefill
            orig, hidden_states = pipeline_prefill(self, input_ids, past_key_values)
            token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
        else:
            pipeline_prefill(self, stage_past_key_values=past_key_values)
        
        for idx in range(max_length):
            # [tree_decoding]
            if self.is_first_stage:
                input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
                    hidden_states,
                    input_ids_ea,
                    self.stage_base_model.lm_head,
                    logits_processor
                )
                seqs_split, lens_split = split_sequence_close_equal_len(
                    draft_tokens,
                    self.total_stage
                )
                tree_decoding_params = (
                    self, past_key_values, seqs_split, lens_split, tree_position_ids, tree_mask, input_ids
                )

            else:
                tree_decoding_params = (self, past_key_values)

            outputs = stage_tree_decoding(*tree_decoding_params)

            if self.is_first_stage:
                logits, hidden_state = outputs
                logits = logits[0, retrieve_indices]

                padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(self.stage_base_model.device)
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)

                candidates = draft_tokens[0, retrieve_indices]
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
                accept_length += 1
                new_token += accept_length
                if log:
                    print(f'{idx}th round, accept_length: {accept_length}')

            # [update_inference_inputs]
            """
            OWNED
            stage_1: input_ids, prev_len, candidates
            middle_stage: 
            stage_n-1: logits, candidates, retrieve_indices, best_candidates, accept_length, select_indices
            NEED
            
            """
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
                    # new_token,
                    hidden_state,
                    sample_p,
                )
            else:
                update_inputs_params = (self, past_key_values_data, current_length_data)

            outputs = update_stage_inference_inputs(*update_inputs_params)
            
            # dist.barrier()
            if self.is_first_stage:
                # input_ids, hidden_states, token, new_token = outputs
                input_ids, hidden_states, token = outputs

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
            prune_first=True,  # [updated] True: prune then send, False: send then prune
            is_llama3=False,
    ):
        """
        pipelined tree decoding with pruning
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # [MODIFIED]: only stage1 has the ea_layer and input_ids
        if temperature > 1e-5 and (self.config.is_first_stage or self.config.is_last_stage):
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        # Initialize the past key and value states
        self.stage_base_model.model.tree_mask = None
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

        # [initialization]
        config = self.config
        device = self.stage_base_model.device
        comm = self.comm
        if self.is_first_stage or self.is_last_stage:
            padding = torch.zeros(1, 1, dtype=torch.long) - 1  # padding -1 to the draft token sequence

        # [initialization] model
        if self.is_first_stage:
            max_length = max_length - self.ea_layer.total_tokens - 10
            assert input_ids is not None, '"input_ids" cannot be None for stage1'
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoiding modifying the input_ids in-place
            # padding = padding.to(input_ids.device)
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

            input_len = input_ids.shape[1]
            reset_tree_mode(self)

        # [initialization] prefill of base model
        output = pipeline_prefill(
            self, input_ids=input_ids, stage_past_key_values=past_key_values
        )
        if config.is_first_stage:
            orig, hidden_state = output
            token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
            new_token = 0

        # get the global accept length (for later pruning)
        global_accept_len = current_length_data[0].item()
        should_stop = torch.tensor(0, dtype=torch.int32)
        new_sampled_token = -1
        
        for idx_spec in range(max_length):

            # print(f'stage{config.stage}, idx_spec={idx_spec}')
            if config.is_first_stage:
                # if idx_spec == 0:
                #     token = gen_token(logits=orig, logits_processor=logits_processor)
                # else:
                #     token = torch.tensor([[new_sampled_token]], dtype=torch.long)
                input_ids_ea = torch.cat((input_ids, token), dim=1)

                # make a draft token tree based on the hidden_state and input_ids
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
                    hidden_state, input_ids_ea, self.stage_base_model.lm_head,
                    logits_processor)

                tree_mask = tree_mask.to(input_ids.device)
                tree_position_ids = tree_position_ids + input_ids.size(-1)

                # split the tree
                draft_tokens_split, lens_split, subseq_ri_cum_depths = token_tree_partition(
                    draft_tokens, retrieve_indices, config.total_stage)
                
                fill_pipeline_params = (
                    self,
                    past_key_values,
                    input_ids,
                    lens_split,
                    draft_tokens,
                    retrieve_indices,
                    tree_mask,
                    tree_position_ids,
                    subseq_ri_cum_depths
                )
            else:
                fill_pipeline_params = (self, past_key_values)

            outputs = fill_pipeline_stages(*fill_pipeline_params)

            if self.is_first_stage:
                sub_hidden_state = outputs
            elif self.is_last_stage:
                sub_hidden_state, lens_split, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, subseq_ri_cum_depths = outputs
            else:  # middle stages
                sub_hidden_state, lens_split, tree_mask, tree_position_ids = outputs

            if self.is_first_stage:
                accept_hidden_states = []

            # continuous verification and pruning
            # inner loop
            for i in range(config.total_stage):
                # pruning
                if lens_split[0] > 0:
                    if config.is_last_stage:
                        # print(f'stage{config.stage} {i}th verification')

                        # [verification]
                        subseq_logits = self.stage_base_model.lm_head(sub_hidden_state)
                        draft_tokens_split = draft_tokens.split(lens_split.tolist(), dim=-1)
                        sub_draft_tokens = torch.cat((draft_tokens_split[0], padding), dim=1)
                        sub_retrieve_indices = get_subtree_retrieve_indices(retrieve_indices, subseq_ri_cum_depths[0])

                        subtree_logits = subseq_logits[0, sub_retrieve_indices]
                        candidates = sub_draft_tokens[0, sub_retrieve_indices]
                        best_candidate, accept_length, sample_p = evaluate_posterior(
                            subtree_logits, candidates, logits_processor
                        )
                        accept_length += 1  # return accept_length - 1 in evaluate_posterior()
                        token = gen_token(prob=sample_p, logits_processor=logits_processor)  # device=cuda

                        cur_draft_depth = subseq_ri_cum_depths[0, best_candidate]
                        if log:
                            print(f'- {i}th turn, accept_len/local_depth: {accept_length}/{cur_draft_depth}')

                        # [local pruning]
                        output = pruning(draft_tokens, retrieve_indices, best_candidate, accept_length, token,
                                        subseq_ri_cum_depths)
                        if not isinstance(output, tuple):  # start new speculation round
                            # print(f'stage{config.stage} truncate')
                            left_indices = output
                            new_sampled_token = token.item()  # send new_sampled_token to the first stage
                            truncate = True  # break after token_pruning()

                        else:
                            draft_tokens, retrieve_indices, left_indices, subseq_ri_cum_depths = output
                            new_sampled_token = -1
                            truncate = False
                        
                        # print(f'stage{config.stage} send {i}th pruning_info')
                        pruning_info_shape = torch.tensor(left_indices.shape, dtype=torch.long) + 2
                        dist.broadcast(pruning_info_shape, src=config.stage)
                        pruning_info = torch.cat((torch.tensor((new_sampled_token, accept_length), dtype=torch.long), left_indices), dim=0).contiguous()
                        # maybe use dist.send() instead
                        dist.broadcast(pruning_info, src=config.stage)  # 3
                        if truncate:
                            lens_split, tree_mask, tree_position_ids = None, None, None  # keep the sub_hidden_state

                    else:
                        pruning_info_shape = torch.zeros(1, dtype=torch.long)
                        dist.broadcast(pruning_info_shape, src=config.total_stage - 1)  # 0
                        pruning_info = torch.zeros(pruning_info_shape.item(), dtype=torch.long)
                        dist.broadcast(pruning_info, src=config.total_stage - 1)

                        new_sampled_token = pruning_info[0].item()
                        accept_length = pruning_info[1].item()
                        left_indices = pruning_info[2:]
                        truncate = new_sampled_token != -1

                        if truncate:
                            sub_hidden_state, lens_split, tree_mask, tree_position_ids = None, None, None, None
                            token = torch.tensor([[new_sampled_token]], dtype=torch.long, device=device)

                    # [global pruning] prune the tokens (kv_cache and hidden_state)
                    # - according to the global_accept_len and the hidden_state_len
                    past_key_values_data, current_length_data, lens_split, sub_hidden_state, tree_mask, tree_position_ids = token_pruning(
                        past_key_values_data,
                        current_length_data,
                        lens_split,
                        sub_hidden_state,
                        tree_mask,
                        tree_position_ids,
                        left_indices,
                        global_accept_len,
                        accept_length,
                        config.stage
                    )

                    global_accept_len += accept_length
                    if config.is_first_stage:
                        new_token += accept_length

                    # start new speculation round
                    if truncate:
                        if config.is_last_stage:
                            comm.sendto(sub_hidden_state.cpu(), config.next_rank)
                            print(f'Stage {config.stage} {i}th turn sendto {config.next_rank} sub_hidden_state: {sub_hidden_state.shape}')
                        if config.is_first_stage:
                            last_hidden_state = comm.recvfrom(config.last_rank, device=self.stage_base_model.device)
                            accept_hidden_states.append(last_hidden_state)

                            accepted_tokens = draft_tokens[:, left_indices]
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                        break

                else:
                    # print(f'stage{config.stage}: no pruning in the {i}th round')
                    lens_split = lens_split[1:]
                    accept_length = 0
                    if config.is_first_stage or config.is_last_stage:  # 但是first stage用不着cum_depths
                        subseq_ri_cum_depths = subseq_ri_cum_depths[1:]

                # stage0 send*1, stage1 send*2, stage2 send*3, stage3 send*4
                # truncate: only the last stage send the sub_hidden_state
                if i < config.stage + 1:
                    if sub_hidden_state.size(-2) > 0:
                        comm.sendto(sub_hidden_state.cpu(), config.next_rank)

                # stage0 recv*4, stage1 recv*1, stage2 recv*2, stage3 recv*3
                # truncate: only stage0 recv the last_hidden_state
                if i < config.last_rank + 1:
                    if config.is_first_stage:
                        hs_len = accept_length
                    else:
                        hs_len = lens_split[config.total_stage - config.stage - 1]
                        tree_pos_ids = tree_position_ids.split(lens_split.tolist(), dim=0)[config.total_stage - config.stage - 1]
                        cum_lens = torch.cumsum(lens_split, dim=0)

                        subseq_idx = config.total_stage - config.stage - 1
                        if config.is_last_stage:
                            # isend_mask.wait()
                            tree_mask_split = tree_mask[..., :cum_lens[0], :cum_lens[0]].contiguous()
                        else:
                            tree_mask_split = tree_mask[..., cum_lens[subseq_idx-1]:cum_lens[subseq_idx], :cum_lens[subseq_idx]].contiguous()
                        self.stage_base_model.model.tree_mask = tree_mask_split
                    
                    if hs_len > 0:  # here
                        last_hidden_state = comm.recvfrom(config.last_rank, device=self.stage_base_model.device)

                        if config.is_first_stage:
                            # print(f'======={i}th stage0_pruning=========')
                            draft_tokens, retrieve_indices, accepted_tokens = first_stage_pruning(
                                left_indices, accept_length, draft_tokens, retrieve_indices
                            )
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                            # new_token += accepted_tokens.size(-1)

                            accept_hidden_states.append(last_hidden_state)
                        else:
                            # print(f'stage{config.stage}: hs_shape:{last_hidden_state.shape}, pos_ids:{tree_pos_ids.shape}')
                            outputs, sub_hidden_state = self(
                                inputs_embeds=last_hidden_state,
                                past_key_values=past_key_values,
                                position_ids=tree_pos_ids
                            )
                    else:  # output is also empty
                        sub_hidden_state = torch.zeros(1, 0, config.hidden_size, dtype=torch.float16)
                else:
                    sub_hidden_state, tree_mask, tree_position_ids = None, None, None

            # end this round of speculative decoding
            if config.is_first_stage:
                hidden_state = torch.cat(accept_hidden_states, dim=-2)  # for draft generation next round
                orig = self.stage_base_model.lm_head(hidden_state[:, -1])
                # stop criteria
                if input_ids is not None and self.tokenizer is not None:
                    if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32)
                    elif self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32)
                    elif new_token > max_new_tokens:
                        should_stop = torch.tensor(1, dtype=torch.int32)
                    elif input_ids.shape[1] > max_length:
                        should_stop = torch.tensor(1, dtype=torch.int32)

                    # print(f'stage{config.stage} send should_stop {should_stop}')
                    dist.broadcast(should_stop, src=0)
                    if should_stop.item() == 1:
                        break
                else:
                    dist.broadcast(should_stop, src=0)
                    if should_stop.item():
                        break
            else:
                # print(f'stage{config.stage} wait should_stop')
                # should_stop = torch.tensor(0, dtype=torch.int32)
                dist.broadcast(should_stop, src=0)  # 1
                if should_stop.item():
                    break

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
        self.stage_base_model.model.tree_mask = None
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

        # [initialization]
        config = self.config
        device = self.stage_base_model.device
        comm = self.comm
        if config.is_first_stage or config.is_last_stage:
            padding = torch.zeros(1, 1, dtype=torch.long) - 1  # padding -1 to the draft token sequence

        # [initialization] model
        if config.is_first_stage:
            max_length = max_length - self.ea_layer.total_tokens - 10
            assert input_ids is not None, '"input_ids" cannot be None for stage1'
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoiding modifying the input_ids in-place
            # padding = padding.to(input_ids.device)
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

            input_len = input_ids.shape[1]
            reset_tree_mode(self)
            new_token = 0

            # prefill
            orig, hidden_state = pipeline_prefill(self, input_ids, past_key_values)
            token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
        else:
            pipeline_prefill(self, stage_past_key_values=past_key_values)

        # get the global accept length (for later pruning)
        global_accept_len = current_length_data[0].item()
        should_stop = torch.tensor(0, dtype=torch.int32)

        turns = 0
        new_sampled_token = -1

        # [continuous speculation] outer loop: start from a new draft tree
        for idx_spec in range(max_length):

            # print(f'stage{config.stage} idx_spec={idx_spec}')
            if config.is_first_stage:
                if log:
                    print(f'{idx_spec}th round [start]')    
                input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
                
                # make a draft token tree based on the hidden_state and input_ids
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.ea_layer.topK_genrate(
                    hidden_state, input_ids_ea, self.stage_base_model.lm_head,
                    logits_processor)

                tree_mask = tree_mask.to(input_ids.device)

                # update tree_position_ids
                tree_position_ids = tree_position_ids + input_ids.size(-1)

                # split the tree
                draft_tokens_split, lens_split, subseq_ri_cum_depths = token_tree_partition(
                    draft_tokens, retrieve_indices, config.total_stage)
                
                fill_pipeline_params = (
                    self,
                    past_key_values,
                    input_ids,
                    lens_split,
                    draft_tokens,
                    retrieve_indices,
                    tree_mask,
                    tree_position_ids,
                    subseq_ri_cum_depths
                )
            else:
                fill_pipeline_params = (self, past_key_values)

            outputs = fill_pipeline_stages(*fill_pipeline_params)

            if self.is_first_stage:
                sub_hidden_state = outputs
                waiting_draft = 0  # 表示未输入pipeline的draft_tokens数量
                accept_length_this_round = 0  # accept_len_per_round

                # 表示last_stage当前拥有了到了多少subseq对应的信息：ri, draft_tokens, subseq_ri_cum_depths
                # first_stage_subseq_process_idx = last_stage_subseq_process_idx = config.total_stage - 1
            elif self.is_last_stage:
                sub_hidden_state, lens_split, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, subseq_ri_cum_depths = outputs
            else:  # middle stages
                sub_hidden_state, lens_split, tree_mask, tree_position_ids = outputs
                
            if self.is_first_stage:
                accept_hidden_states = []

            i = -1
            
            # inner loop for continuous speculation: [exit] the loop when truncated
            while True:
                i += 1
                # pruning
                if lens_split[0] > 0:
                    if config.is_last_stage:
                        # [verification]
                        # print(f'stage{config.stage} {i}th verification')
                        subseq_logits = self.stage_base_model.lm_head(sub_hidden_state)
                        draft_tokens_split = draft_tokens.split(lens_split.tolist(), dim=1)
                        sub_draft_tokens = torch.cat((draft_tokens_split[0], padding), dim=1)
                        sub_retrieve_indices = get_subtree_retrieve_indices(retrieve_indices, subseq_ri_cum_depths[0])

                        subseq_ri_cum_depths = subseq_ri_cum_depths[1:]  # remove the first subseq

                        subtree_logits = subseq_logits[0, sub_retrieve_indices]
                        candidates = sub_draft_tokens[0, sub_retrieve_indices]
                        best_candidate, accept_length, sample_p = evaluate_posterior(
                            subtree_logits, candidates, logits_processor
                        )
                        accept_length += 1
                        # print(f'last stage {idx_spec}th round {i}th turn: {torch.topk(sample_p, k=10, dim=-1)}')
                        token = gen_token(prob=sample_p, logits_processor=logits_processor)  # device=cuda
                        # token = torch.multinomial(sample_p, num_samples=1)

                        cur_draft_depth = subseq_ri_cum_depths[0, best_candidate]
                        if log:
                            print(f'- {i}th turn, accept_len/local_depth: {accept_length}/{cur_draft_depth}')
                        
                        # [local pruning]
                        output = pruning(draft_tokens, retrieve_indices, best_candidate, accept_length, token, subseq_ri_cum_depths)
                        if not isinstance(output, tuple):  # start new speculation round
                            left_indices = output
                            new_sampled_token = token.item()
                            truncate = True
                            if log:
                                print(f'- {i}th turn truncate')
                        
                        else:
                            draft_tokens, retrieve_indices, left_indices, subseq_ri_cum_depths = output
                            new_sampled_token = -1
                            truncate = False

                        pruning_info = torch.cat((torch.tensor((new_sampled_token, accept_length), dtype=torch.long), left_indices), dim=0).contiguous()
                        if not truncate:  # not truncate: async pruning info broadcast
                            broadcast_pruning_info_task = comm.executor.submit(
                                comm.broadcast_send,
                                pruning_info
                            )
                        else:  # truncate: sync pruning info broadcast
                            comm.broadcast_send(pruning_info)
                            lens_split = tree_mask = tree_position_ids = None
                        
                    else:
                        pruning_info = comm.broadcast_recv(config.total_stage - 1)

                        new_sampled_token = pruning_info[0].item()
                        accept_length = pruning_info[1].item()
                        left_indices = pruning_info[2:]
                        
                        truncate = new_sampled_token != -1
                        if truncate:
                            sub_hidden_state = lens_split = tree_mask = tree_position_ids = None
                            if config.is_first_stage:
                                token = torch.tensor([[new_sampled_token]], dtype=torch.long, device=input_ids.device)

                    if config.is_first_stage:
                        # [first stage pruning]
                        # prune subseq_ri_cum_depths for continuous speculation
                        if not truncate:
                            draft_tokens, retrieve_indices, accepted_tokens, subseq_ri_cum_depths, left_indices = first_stage_pruning(
                                    left_indices, accept_length, draft_tokens, retrieve_indices, subseq_ri_cum_depths
                            )
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                            # first stage may have unhandled tokens


                    # [global pruning] prune the tokens (kv_cache and hidden_state)
                    # - according to the global_accept_len and the hidden_state_len
                    past_key_values_data, current_length_data, lens_split, sub_hidden_state, tree_mask, tree_position_ids = token_pruning(
                        past_key_values_data,
                        current_length_data,
                        lens_split,
                        sub_hidden_state,
                        tree_mask,
                        tree_position_ids,
                        left_indices,
                        global_accept_len,
                        accept_length,
                        config.stage
                    )

                    # 剪枝完得到waiting_draft
                    if config.is_first_stage:
                        if not truncate:
                            waiting_draft = (draft_tokens.size(-1) - torch.sum(lens_split)).item()
                            # print(f'stage{config.stage} {i}th waiting_draft: {waiting_draft} after pruning')

                    global_accept_len += accept_length
                    if config.is_first_stage:
                        
                        new_token += accept_length
                        accept_length_this_round += accept_length

                    # start new speculation round
                    if truncate:
                        if config.is_last_stage:
                            comm.sendto(sub_hidden_state.cpu(), config.next_rank)
                        if config.is_first_stage:
                            i -= 1  # truncate这一轮循环没有实际计算
                            last_hidden_state = comm.recvfrom(config.total_stage - 1, device=self.stage_base_model.device)
                            accept_hidden_states.append(last_hidden_state)

                            accepted_tokens = draft_tokens[:, left_indices]
                            input_ids = torch.cat((input_ids, accepted_tokens), dim=-1)
                        break

                else:
                    lens_split = lens_split[1:]
                    accept_length = 0
                    if config.is_first_stage or config.is_last_stage:  # [update]: drop the first row also for the first stage
                        subseq_ri_cum_depths = subseq_ri_cum_depths[1:]
                # [the above part is exactly the same as eagenerate_pruned_pipeline()]
                        
                # 此时：
                # 1. 所有stage都要发送sub_hidden_state给下一个stage
                # - last stage发的是accept_hidden_state
                # - first stage除了第一次发送的是原本树的部分，后续所有发送的都是树的延申部分
                # 2. 除了stage都要从上一个stage接收last_hidden_state
                # - first stage接收的是accept_hidden_state
                # 3. first stage在循环中
                #   - 先启动异步发送剪枝后的sub_hidden_state
                #   - 再启动异步接收accept_hidden_state
                #   - 然后开始树的延申
                # 4. 第一轮循环之后，所有stage需要异步接收expansion_info

 
                if sub_hidden_state.size(-2) > 0:
                    # isend_task = dist.isend(sub_hidden_state.cpu(), dst=config.next_rank)
                    comm.sendto(sub_hidden_state.cpu(), config.next_rank)
                else:
                    isend_task = None

                # 2. first_stage: irecv(), others: recv()

                # get size of last_hidden_state
                if config.is_first_stage:
                    hs_len = accept_length
                    received = False  # recv [grow or not grow]
                else:
                    hs_len = lens_split[config.total_stage - config.stage - 1]

                # last_hidden_state = torch.zeros((1, hs_len, config.hidden_size), dtype=torch.float16)
                
                # recv last_hidden_state
                if config.is_first_stage:
                    # tree expansion and forward
                    cur_draft_depth = (retrieve_indices != -1).sum(dim=1).max()
                    # cur_draft_size = draft_tokens.size(-1)
                    # print(f'cur_draft_depth: {cur_draft_depth}')
                    # print(f'cur_draft_size: {cur_draft_size}')

                    if waiting_draft < 32:  # todo: set proper expand_condition
                        # expand the tree for more waiting_draft    

                        # new_token不需要通过hidden_state生成，直接根据剪枝的token生成，新token就是剪枝后新树的根节点
                        input_ids_ea = torch.cat((input_ids, draft_tokens[:, :1]), dim=-1)

                        if hs_len > 0:
                            last_hidden_state = comm.recvfrom(config.last_rank, device=self.stage_base_model.device)
                            accept_hidden_states.append(last_hidden_state)
                            received = True  # says last_hidden_state has been received
                        accepted_hidden_state = torch.cat(accept_hidden_states, dim=-2)
                        accept_hidden_states = []

                        # print(f'stage{config.stage} {i}th input_ids_ea: {input_ids_ea.shape}, accepted_hidden_state: {accepted_hidden_state.shape}')

                        draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2 = self.ea_layer.topK_genrate(
                            accepted_hidden_state,
                            input_ids_ea,
                            self.stage_base_model.lm_head,
                            logits_processor,
                            depth=max(cur_draft_depth + 3, 6),  # todo: test best tree settings
                            total_tokens=80
                        )  # get a little more appended tokens
                        tree_position_ids2 = tree_position_ids2 + input_ids.size(-1)

                        # print(tree_position_ids[0], tree_position_ids2[0])
                        assert tree_position_ids[0] == tree_position_ids2[0], 'Should starts from same pos_id'

                        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, lens_split, subseq_ri_cum_depths = merge_two_tree(
                            (draft_tokens, retrieve_indices, tree_mask, tree_position_ids),
                            (draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2),
                            lens_split,
                            subseq_ri_cum_depths
                        )

                        waiting_draft = lens_split[-1].item()

                        # broadcast expand_info
                        # - (global) middle stages: appended tree_pos_ids and tree_mask
                        # appended_draft_len
                        appended_draft_len = min(waiting_draft, 32)  # todo: set set_subseq_len = 16?
                        lens_split[-1] = appended_draft_len
                        # print(f'stage{config.stage} {i}th lens_split: {lens_split} before broadcast')

                    else:
                        appended_draft_len = min(waiting_draft, 32)
                        # print(f'stage{config.stage} {i}th appended_draft_len: {appended_draft_len}, waiting_draft: {waiting_draft}')
                        lens_split = torch.cat((lens_split, torch.tensor([appended_draft_len], dtype=torch.long)))
                    
                    # print(f'waiting_draft: {waiting_draft}, appended_draft_len: {appended_draft_len}')
                    waiting_draft -= appended_draft_len
                    # print(f'stage{config.stage} {i}th waiting_draft: {waiting_draft}, appended_draft_len: {appended_draft_len}')
                    existing_draft_len = torch.sum(lens_split[:-1])
                    input_draft_end_idx = existing_draft_len + appended_draft_len
                    # print(f'stage{config.stage} {i}th existing_draft_len: {existing_draft_len}, input_draft_end_idx: {input_draft_end_idx}')

                    # 计算最新输入subseq的subseq_ri_cum_depths
                    cur_subseq_ri_cum_depth = subseq_ri_cum_depths[-1].clone()
                    # add -1 to last layer of retrieve_indices
                    retrieve_indices_filled = torch.cat((retrieve_indices, torch.full((retrieve_indices.size(0), 1), -1, dtype=torch.long)), dim=1)

                    for j in range(existing_draft_len, input_draft_end_idx):
                        row_indices = torch.arange(retrieve_indices.size(0), dtype=torch.long)
                        cum_ri_leaves = retrieve_indices_filled[row_indices, cur_subseq_ri_cum_depth]
                        cur_subseq_ri_cum_depth[cum_ri_leaves == j] += 1
                    subseq_ri_cum_depths = torch.cat((subseq_ri_cum_depths, cur_subseq_ri_cum_depth.unsqueeze(0)), dim=0)

                    appended_draft_tokens = draft_tokens[:, existing_draft_len:input_draft_end_idx]
                    appended_tree_position_ids = tree_position_ids[existing_draft_len:input_draft_end_idx]
                    appended_tree_mask = tree_mask[..., existing_draft_len:input_draft_end_idx, :input_draft_end_idx]
                    # print(f'===Send Appended Tree===')
                    # print(f'-- draft_tokens: {draft_tokens[:, :input_draft_end_idx]}')
                    # print(f'-- tree_mask: {tree_mask.shape}; appended_tree_position_ids: {appended_tree_position_ids.shape}; appended_tree_mask: {appended_tree_mask.shape}')
                    broadcast_tree_info_task = comm.executor.submit(
                        comm.broadcast_tree_info,
                        lens_split,
                        appended_tree_position_ids,
                        appended_tree_mask,
                        appended_draft_tokens,
                        retrieve_indices,
                        subseq_ri_cum_depths,
                        appended=True
                    )

                    tree_mask_split = tree_mask[..., existing_draft_len:input_draft_end_idx, :input_draft_end_idx]
                    self.stage_base_model.model.tree_mask = tree_mask_split

                    outputs, sub_hidden_state = self(
                        input_ids=appended_draft_tokens,
                        past_key_values=past_key_values,
                        position_ids=appended_tree_position_ids
                    )
                    if hs_len > 0 and not received:
                        last_hidden_state = comm.recvfrom(config.last_rank, device=device)
                        accept_hidden_states.append(last_hidden_state)
                    broadcast_tree_info_task.result()

                else:
                    # get tree_mask and tree_position_ids
                    if config.is_last_stage:
                        broadcast_pruning_info_task.result()

                    broadcast_tree_info_task = comm.executor.submit(comm.broadcast_tree_info, appended=True)
                    cum_lens = torch.cumsum(lens_split, dim=0)
                    subseq_idx = config.total_stage - config.stage - 1
                    if config.is_last_stage:
                        tree_pos_ids = tree_position_ids[:cum_lens[subseq_idx]]
                        tree_mask_split = tree_mask[..., :cum_lens[0], :cum_lens[0]].contiguous()
                    else:
                        tree_pos_ids = tree_position_ids[cum_lens[subseq_idx-1] : cum_lens[subseq_idx]]
                        tree_mask_split = tree_mask[..., cum_lens[subseq_idx-1]:cum_lens[subseq_idx], :cum_lens[subseq_idx]].contiguous()
                    self.stage_base_model.model.tree_mask = tree_mask_split

                    # other stages sync recv and forward
                    if hs_len > 0:

                        last_hidden_state = comm.recvfrom(config.last_rank, device=device)
                        outputs, sub_hidden_state = self(  # forward
                            inputs_embeds=last_hidden_state,
                            past_key_values=past_key_values,
                            position_ids=tree_pos_ids
                        )
                    else:
                        sub_hidden_state = torch.zeros(1, 0, config.hidden_size, dtype=torch.float16)

                    # recv global expand_info
                    expand_info = broadcast_tree_info_task.result()
                    if not config.is_last_stage:
                        lens_split, appended_tree_position_ids, appended_tree_mask = expand_info
                    else:
                        lens_split, appended_tree_position_ids, appended_tree_mask, appended_draft_tokens, retrieve_indices, subseq_ri_cum_depths = expand_info
                        draft_tokens = torch.cat((draft_tokens, appended_draft_tokens), dim=-1)

                    appended_draft_len = lens_split[-1]
                    tree_position_ids = torch.cat((tree_position_ids, appended_tree_position_ids.to(tree_position_ids.device)), dim=0)
                    tree_mask = F.pad(tree_mask, (0, appended_draft_len), value=0)
                    tree_mask = torch.cat((tree_mask, appended_tree_mask.to(tree_mask.device)), dim=-2)
                    
            # end this round of speculative decoding
            # todo: maybe end in the continous speculation loop
            if config.is_first_stage:
                turns += i+1
                if log:
                    print(f'{idx_spec}th round [end] accept_length={accept_length_this_round} turns={i+1}')

                # print(f'accept_hidden_states: {[hs.size(-2) for hs in accept_hidden_states]}')
                hidden_state = torch.cat(accept_hidden_states, dim=-2)  # for draft generation next round
                orig = self.stage_base_model.lm_head(hidden_state[:, -1])
                # stop criteria
                if input_ids is not None and self.tokenizer is not None:
                    if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32)
                    elif self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32)
                    elif new_token > max_new_tokens:
                        should_stop = torch.tensor(1, dtype=torch.int32)
                    elif input_ids.shape[1] > max_length:
                        should_stop = torch.tensor(1, dtype=torch.int32)

                    dist.broadcast(should_stop, src=0)
                    if should_stop.item() == 1:
                        break
                else:
                    dist.broadcast(should_stop, src=0)
                    if should_stop.item():
                        break
            else:
                dist.broadcast(should_stop, src=0)  # 1
                if should_stop.item():
                    break

        if self.is_first_stage:
            if not log:
                return input_ids
            else:
                return input_ids, new_token, idx_spec, turns



