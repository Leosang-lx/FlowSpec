import json
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer
from eagle.utils import *
# from eagle.kv_cache import initialize_past_key_values
from tp.tp_kv_cache import initialize_past_key_values
from tp_modeling_llama import TPLlamaForCausalLM
from stage_ea_config import StageEaConfig
from comm.comm_handler import CommHandler
from accelerate import init_empty_weights
from contextlib import nullcontext
from config.run_config import config as run_config
from eagle.cnets import Model  # [ADDED] Draft model loading
from huggingface_hub import hf_hub_download
from tools.communicator import *
from pipeline_utils import *
import os

class TPEaModel(nn.Module):
    def __init__(self, tp_base_model, model_path, config, ea_draft_model=None, init_comm=True):
        super().__init__()
        
        self.tp_base_model = tp_base_model
        self.config = config
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        stage = config.stage
        self.is_draft_stage = config.stage == 0 and config.has_draft_model

        if self.is_draft_stage:
            assert ea_draft_model is not None
            assert isinstance(ea_draft_model, Model)
            self.ea_layer = ea_draft_model
            self.ea_layer.init_tree()

        if init_comm:
            self.comm = CommHandler(rank=config.stage, world_size=config.total_stage, enable_async_send_recv=False, timeout=run_config.timeout)
            # print(f"init comm handler")
            self.comm.init_PG()
            subgroup_ranks = [1,2,3,4]
            self.tp_group = self.comm.new_group(subgroup_ranks)
            # self.comm.start_threads()
            if run_config.hardware == "jetson" and not run_config.set_network:
                self.comm.reset_traffic()       
            if run_config.hardware == "jetson" and run_config.set_network:
                self.comm.traffic_control(run_config.rate_mbps, run_config.delay_ms)
            dist.barrier()
            
    @classmethod
    def from_pretrained(cls, 
                        tp_base_model_path, 
                        ea_model_path=None, 
                        total_token=59, 
                        depth=5, 
                        top_k=10, 
                        threshold=1.0, 
                        init_comm=True, 
                        **kwargs):
        model_config = StageEaConfig.from_pretrained(tp_base_model_path)
        tp_base_model = TPLlamaForCausalLM.from_pretrained(
            tp_base_model_path, tp_rank=model_config.stage, tp_size=model_config.total_stage-1, **kwargs
        )

        draft_model = None
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
                                                map_location=tp_base_model.device)
            except:
                from safetensors.torch import load_file
                load_model_path = os.path.join(ea_model_path, "model.safetensors")
                if not os.path.exists(load_model_path):
                    load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
                ea_layer_state_dict = load_file(load_model_path,
                                                device=tp_base_model.device)
                
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
            device = tp_base_model.lm_head.weight.device
            # print(f"stage_base_model.lm_head.weight.device={stage_base_model.lm_head.weight.device}")
            # print(f"device={device}")
            if device != tp_base_model.lm_head.weight.device:
                ea_layer.diff_device = True
                if not low_memory:
                    ea_layer.headweight = tp_base_model.lm_head.weight.clone().to(device)
                else:
                    ea_layer.layer_device = device

            else:
                ea_layer.diff_device = False
            ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
            
            del ea_layer_state_dict, ea_config, con
            torch.cuda.empty_cache()
            
            ea_layer.to(tp_base_model.dtype).to(device)
            # print(f"ea_layer.embed_tokens.weight.device={ea_layer.embed_tokens.weight.device}")
        else:
            ea_layer = None

        return cls(tp_base_model, tp_base_model_path, model_config, ea_draft_model=ea_layer, init_comm=init_comm)

    @torch.no_grad()
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None, 
                output_orig=False,
                prof=None,
                tp_group=None):
        outputs = self.tp_base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            tp_group=tp_group,
        )
        
        hidden_states = outputs[0]
        if output_orig:
            orig = self.tp_base_model.lm_head(hidden_states)
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def tp_generate(
        self,
        input_ids=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        profiler=None,
    ):
        if temperature > 1e-5:  # 先让全部stage都有，怕出bug
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
            
        if self.is_draft_stage:
            self.ea_layer.reset_kv()
        else:
            self.tp_base_model.model.tree_mask = None
            
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
                ) = initialize_past_key_values(self.tp_base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data
        
        should_stop = torch.tensor(0, dtype=torch.int32)  # for break the outer loop
        
        config = self.config
        device = self.tp_base_model.device
        comm = self.comm
        reset_tree_mode(self)
        
        # print(f'rank: {config.stage} start prefill')
        if self.is_draft_stage:
            # [update] get input_ids from the first stage
            input_ids = input_ids.clone()
            input_len = input_ids.shape[1]
            # orig, hidden_state = chunked_prefill(self, input_ids=input_ids, prof=profiler)
            orig, hidden_state = tp_prefill(self, input_ids=input_ids, prof=profiler)
            
            new_token = 0
            token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
            # print(f'rank: {config.stage} token: {token}')
            global skip_count 
            skip_count = 0

        else:
            # chunked_prefill(self, stage_past_key_values=past_key_values, prof=profiler)
            tp_prefill(self, past_key_values=past_key_values, prof=profiler)
        
        # print(f'rank: {config.stage} finished prefill')
        
        should_stop = torch.tensor(0, dtype=torch.int32)
        turns_cnt = 0
        # new_sampled_token = -1
        # no kv_cache for the draft stage
        kv_cache=(past_key_values, past_key_values_data, current_length_data) if not self.is_draft_stage else None

        for idx_spec in range(max_length):
            # if config.stage == 0:
            #     print(f'rank: {config.stage} start idx_spec: {idx_spec}')
            if config.is_draft_stage:
                input_ids_ea = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, last_ea_state = self.ea_layer.topK_genrate(
                    hidden_state,
                    input_ids_ea,
                    self.tp_base_model.lm_head,
                    logits_processor,
                )
                tree_position_ids = tree_position_ids + input_ids.size(-1)
                #send to all other tp stages
                comm.broadcast_send(draft_tokens)
                comm.broadcast_send(tree_position_ids)
                comm.broadcast_send(tree_mask)
                
                #recv from all other tp stages
                # print(f'rank: {config.stage} was at barrier at idx_spec: {idx_spec}')
                # dist.barrier()
                # hidden_state_verified = comm.gather_recv(config.total_stage)
                # hidden_state_verified = hidden_state_verified[0].to(device)
                hidden_state_verified = comm.recv_tensor(src_rank=4).to(device)
                logits = self.tp_base_model.lm_head(hidden_state_verified)
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
            else:
                past_key_values, past_key_values_data, current_length_data = kv_cache
                # recv from draft stage
                input_tensor = comm.broadcast_recv(src_rank=0).to(device)
                tree_position_ids = comm.broadcast_recv(src_rank=0).to(device)
                tree_mask = comm.broadcast_recv(src_rank=0).to(device)
                self.tp_base_model.model.tree_mask = tree_mask
                outputs, hidden_state = self(
                    input_ids=input_tensor,
                    past_key_values=past_key_values,
                    position_ids=tree_position_ids,
                    tp_group=self.tp_group,
                )
                # send to draft stage
                # print(f'rank: {config.stage} was at barrier at idx_spec: {idx_spec}')
                # dist.barrier()
                # comm.gather_send(hidden_state)
                if config.stage == 4:
                    comm.send_tensor(hidden_state.to(comm.comm_device), dst_rank=0)
                
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

            outputs = update_tp_inference_inputs(*update_inputs_params)
            if self.is_draft_stage:
                input_ids, hidden_state, token = outputs
                # assert accept_length == len(input_ids[0, input_len:]) - new_token, f'accept_length: {accept_length} != len(input_ids[0, input_len:]) - new_token: {len(input_ids[0, input_len:]) - new_token}'
                new_token += accept_length
                # if log:
                    # print(f'{idx_spec}th round, accept_length: {accept_length} in {turns} turns')
                    # turns_cnt += turns

                if input_ids is not None and self.tokenizer is not None:
                    if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.tp_base_model.device)
                    elif new_token > max_new_tokens:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.tp_base_model.device)
                    elif input_ids.shape[1] > max_length:
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.tp_base_model.device)
                    
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

@torch.no_grad()
def update_tp_inference_inputs(
    model,
    past_key_values_data_list,
    current_length_data,
    logits_processor=None,
    input_ids=None,
    candidates=None,
    best_candidate=None,
    accept_length=None,
    retrieve_indices=None,
    # new_token=None,
    hidden_state_new=None,
    sample_p=None,
):  
    if model.is_draft_stage:
        prev_input_len = torch.tensor(input_ids.shape[1], dtype=torch.long)
        # dist.broadcast(prev_input_len, src=0)
        broadcast(prev_input_len, src=0)
        select_indices = (retrieve_indices[best_candidate, :accept_length] + prev_input_len)
        broadcast(select_indices, src=0)
    
        # Append the tokens from the best candidate to the input sequence
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length].to(input_ids.device)], dim=-1
        )
    else:
        prev_input_len = broadcast(src=0, data_type=torch.int64, shape_length=0)
        select_indices = broadcast(src=0, data_type=torch.int64, shape_length=1)

    if model.is_draft_stage:
        retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length]
        token = gen_token(prob=sample_p, logits_processor=logits_processor)[None]
        return input_ids, accept_hidden_state_new, token
    
    # Update the past key values based on the selected tokens
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])
    
def tp_prefill(
    tp_model,
    input_ids=None,
    past_key_values=None,
    prof=None
):
    config = tp_model.config
    device = tp_model.tp_base_model.device
    comm = tp_model.comm  # update
    # print(f'rank: {config.stage} start prefill inside') # [update] remove this line
    if config.is_draft_stage:
        comm.broadcast_send(input_ids)
        # print(f'rank: {config.stage} was at barrier')
        # dist.barrier()
        # hidden_state = comm.gather_recv(config.total_stage)
        # hidden_state = hidden_state[0].to(device)
        hidden_state = comm.recv_tensor(src_rank=4).to(device)
        # print(f"rank: {config.stage} hidden_state: {hidden_state}")
        orig = tp_model.tp_base_model.lm_head(hidden_state)
        # print(f"rank: {config.stage} orig: {orig}")
        return orig, hidden_state
    else:
        input_ids = comm.broadcast_recv(src_rank=0).to(device)
        # print(f'rank: {config.stage} input_ids.shape: {input_ids.shape}')
        _, hidden_state = tp_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            tp_group=tp_model.tp_group,
        )
        # print(f'rank: {config.stage} was at barrier')
        # print(f'hidden_state.shape: {hidden_state.shape}')
        # print(f'hidden_state: {hidden_state}')
        # dist.barrier()
        # comm.gather_send(hidden_state)
        if config.stage == 4:
            comm.send_tensor(hidden_state.to(comm.comm_device), dst_rank=0)
       
            
        
        
def reset_tree_mode(
        model,
):
    model.tp_base_model.model.tree_mask = None
    model.tp_base_model.model.tree_mode = None