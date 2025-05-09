# reference: https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/utils.py - def evaluate_posterior()
# codes with comment "EAGLE" are original EAGLE codes

import random
import numpy as np
import time
import torch
import torch.nn.functional as F
from queue import deque
from typing import Union, Iterable, List
import torch.distributed as dist
from datetime import datetime
# from memory_profiler import profile
# from stage_ea_model import StageEaModel
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from tools.communicator import *
from typing import Tuple
from contextlib import nullcontext
TOPK = 10  # topk for sparse tree

def get_time_str():
    now = datetime.now()
    return f"{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}"


def calculate_model_size_with_buffers(model):
    total_memory = 0
    
    for key, tensor in model.state_dict().items():
        if 'lm_head' not in key:
            total_memory += tensor.element_size() * tensor.numel()
    
    # transform to MB
    memory_size_mb = total_memory / (1024 ** 2)
    return memory_size_mb

# EAGLE
class Timer:
    def __init__(self, name, gpu=False):
        self.name = name

    def __enter__(self):
        if self.gpu:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'==== {self.name} {elapsed} seconds ====')


# EAGLE
def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# EAGLE
def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


# EAGLE
def reset_tree_mode(
        model,
):
    model.stage_base_model.model.tree_mask = None
    model.stage_base_model.model.tree_mode = None


# EAGLE
# def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
#     """
#     Resets the current lengths in the passed key-values to zero.
#
#     This function is designed to be used during the evaluation of a baseline model.
#     It iterates through each layer's key-values and sets their current lengths to zero,
#     effectively resetting their state.
#
#     Args:
#     - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.
#
#     Returns:
#     - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
#     """
#     for i in range(len(passed_key_values)):
#         for j in range(2):
#             passed_key_values[i][j].current_length.fill_(0)
#     return passed_key_values

def split_close_equal(total_size, n) -> list:
    assert total_size > n > 0
    base_size = total_size // n
    reminder = total_size % n
    if reminder == 0:
        return [base_size for _ in range(n)]
    else:
        # leave the smaller one on the front
        split_lens = [base_size if i >= reminder else base_size + 1 for i in range(n)]
        split_lens.reverse()
        return split_lens


def split_sequence_close_equal_len(sequence: torch.Tensor, split_cnt: Union[int, Iterable[int], list]): #  tuple is for tree partition
    if len(sequence.shape) <= 2:
        seq_len = sequence.size(-1)
    else:
        raise ValueError('Sequence for splitting can not have a batch_size larger than 2.')

    if isinstance(split_cnt, int):
        split_lens = split_close_equal(seq_len, split_cnt)
    else:
        split_lens = split_cnt

    assert sum(split_lens) == seq_len
    split_seqs = sequence.split(tuple(split_lens), dim=-1)
    split_lens = torch.tensor(split_lens, dtype=torch.long)
    return split_seqs, split_lens


# [ADD] logits processor
def gen_token(logits=None, prob=None, logits_processor=None):
    if logits_processor is not None:
        if logits is not None:
            logits = logits_processor(None, logits)
            prob = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(prob, 1)

    else:
        if logits is not None:
            prob = logits
        token = torch.argmax(prob, dim=-1)
        token = token[None]

    return token


def pipeline_prefill(
        stage_model,
        input_ids=None,
        stage_past_key_values=None,
):
    config = stage_model.config
    stage_base_model = stage_model.stage_base_model
    device = stage_base_model.device
    total_stage = config.total_stage
    comm = stage_model.comm  # update
    
    if config.is_first_stage:
        # todo: split strategy depends on the device performance and the input length
        if input_ids.size(-1) > 64:  # pipelined prefill, split the input_ids when long enough
            seq_splits, lens_split = split_sequence_close_equal_len(input_ids, config.n_split)
            # hidden_state_splits = []  # save the hidden_state for later transmission
        else:
            lens_split = torch.zeros(config.n_split, dtype=torch.long)
            lens_split[0] = input_ids.size(-1)
            seq_splits = (input_ids,)
    # else:
    #     lens_split = torch.zeros(total_stage, dtype=torch.long)

    # dist.broadcast(lens_split, src=1)
        lens_split = lens_split[lens_split > 0]
        
    if config.is_first_stage:
        comm.sendto(input_ids, dst_rank=0)
        
    # [update]: draft stage recv base_hidden_state when prefilling
    if config.is_draft_stage:  # assume 4 subseqs
        input_ids = comm.recvfrom(src_rank=1, device=device)
        # hidden_state_splits = [torch.zeros(input_ids.shape[0], lens_split[i], config.hidden_size, device=device) for i in range(lens_split.size(-1))]
        hidden_state_splits = []
        orig = ()
        for i in range(config.n_split):
            hidden_state_split = comm.recvfrom(config.last_rank, device=device).to(device)
            hidden_state_splits.append(hidden_state_split)
            orig = orig + (stage_base_model.lm_head(hidden_state_split),)

        hidden_state = torch.concat(hidden_state_splits, dim=-2)
        orig = torch.concat(orig, dim=-2)

        return input_ids, orig, hidden_state

    for i in range(config.n_split):
        if config.is_first_stage:
            outputs, sub_hidden_state = stage_model(
                input_ids=seq_splits[i],
                past_key_values=stage_past_key_values,
            )
            comm.sendto(sub_hidden_state.cpu(), config.next_rank)
        else:
            last_hidden_state = comm.recvfrom(config.last_rank, device=device)

            # middle stage
            outputs = stage_base_model.model(
                inputs_embeds=last_hidden_state,
                past_key_values=stage_past_key_values,
            )
            sub_hidden_state = outputs[0]
            comm.sendto(sub_hidden_state.cpu(), config.next_rank)
            
def pipeline_prefill_new(
        stage_model,
        input_ids=None,
        stage_past_key_values=None,
        prof=None
):
    config = stage_model.config
    device = stage_model.stage_base_model.device
    comm = stage_model.comm  # update
        
    if config.is_draft_stage:  # assume 4 subseqs
        seq_splits, _ = split_sequence_close_equal_len(input_ids, config.n_split)
        
        for subseq in seq_splits:
            comm.sendto(subseq.cpu(), config.next_rank)
            
        hidden_state_splits = []
        orig = ()
        for i in range(config.n_split):
            with prof.time_context(f"Stage {config.stage}: recv from last stage", cpu=True) if prof is not None else nullcontext():
                hidden_state_split = comm.recvfrom(config.last_rank, device=device).to(device)
            hidden_state_splits.append(hidden_state_split)
            orig = orig + (stage_model.stage_base_model.lm_head(hidden_state_split),)

        hidden_state = torch.concat(hidden_state_splits, dim=-2)
        orig = torch.concat(orig, dim=-2)

        return orig, hidden_state

    for i in range(config.total_stage-1):
        with prof.time_context(f"Stage {config.stage}: recv from last stage", cpu=True) if prof is not None else nullcontext():
            last_hidden_state = comm.recvfrom(config.last_rank, device=device)

        with prof.time_context(f"Stage {config.stage}: prefill forward", cpu=False) if prof is not None else nullcontext():
            if config.is_first_stage:
                _, sub_hidden_state = stage_model(
                    input_ids=last_hidden_state,
                    past_key_values=stage_past_key_values,
                )
                comm.sendto(sub_hidden_state.cpu(), config.next_rank)
            else:
                _, sub_hidden_state = stage_model(
                    inputs_embeds=last_hidden_state,
                    past_key_values=stage_past_key_values,
                )
                comm.sendto(sub_hidden_state.cpu(), config.next_rank)

    

def prefill_pipeline0(stage_model, stage_past_key_values=None, input_ids = None):
    # print(f"stage_model.stage: {stage_model.stage}, layer_range: {stage_model.stage_base_model.model.config.layer_range}")
    if stage_model.is_first_stage:
        outputs, hidden_states = stage_model(
            input_ids=input_ids,
            past_key_values=stage_past_key_values,
        )
        send(hidden_states, dst=1)
    else:
        inputs_embeds = recv(src=stage_model.stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
        if stage_model.is_last_stage:
            outputs, orig, hidden_states = stage_model(
                inputs_embeds=inputs_embeds,
                past_key_values=stage_past_key_values,
                output_orig=True,
            )
            send(orig, dst=0)
            send(hidden_states, dst=0)
        else:
            outputs, hidden_states = stage_model(
                inputs_embeds=inputs_embeds,
                past_key_values=stage_past_key_values,
            )
            send(hidden_states, dst=stage_model.stage + 1)
            
    if stage_model.is_first_stage:
        orig = recv(src=stage_model.total_stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
        hidden_states = recv(src=stage_model.total_stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
        return orig, hidden_states

# [MODIFIED] from initialize_tree()
def initialize_tree_pipeline(stage_model, past_key_values, logits_processor=None, input_ids=None):
    if stage_model.is_first_stage:
        orig, hidden_states = pipeline_prefill(stage_model, input_ids, past_key_values)
        
        token = gen_token(logits=orig[:, -1], logits_processor=logits_processor)
        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = stage_model.ea_layer.topK_genrate(
            hidden_states,
            input_ids,
            stage_model.stage_base_model.lm_head,
            logits_processor
        )
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token
    else:
        pipeline_prefill(stage_model, stage_past_key_values=past_key_values)


# [MODIFIED] from tree_decoding()
def stage_tree_decoding(
        stage_model,
        stage_past_key_values=None,
        draft_seqs_split=None,
        lens_split=None,
        tree_pos_ids=None,
        tree_mask=None,
        input_ids=None,
        prof=None
):
    """
    pipelined tree decoding for verification
    :param stage_model: necessary
    :param stage_past_key_values: necessary
    :param draft_seqs_split: only necessary for stage1
    :param tree_pos_ids_split: only necessary for stage1
    :param lens_split: only necessary for stage1
    :param input_ids: only necessary for stage1
    :param retrieve_indices: only necessary for stage(-1)
    :return: return result on stage1
    """
    # [step1] prepare necessary data to all devices
    config = stage_model.config
    device = stage_model.stage_base_model.device
    comm = stage_model.comm
    profiler = prof
    if config.is_draft_stage:
        tree_pos_ids = tree_pos_ids + input_ids.size(-1)  # add the input length to the tree position ids
        broadcast_tree_info_global_task = comm.executor.submit(
            comm.broadcast_tree_info_global,
            lens_split,
            tree_pos_ids,
            tree_mask,
            appended=False
        )
        tree_logits_split = ()
        hidden_state_splits = []
        
        for split_draft_seqs in draft_seqs_split:
            comm.sendto(split_draft_seqs, dst_rank=config.next_rank)
            
        broadcast_tree_info_global_task.result()
    else:
        lens_split, tree_pos_ids, tree_mask = comm.broadcast_tree_info_global(appended=False)
    
    if stage_model.is_draft_stage:
        for i in range(lens_split.size(-1)):
            hidden_state_split = comm.recvfrom(config.last_rank, device=device)
            hidden_state_splits.append(hidden_state_split.to(device))
            tree_logits_split = tree_logits_split + (stage_model.stage_base_model.lm_head(hidden_state_splits[i]),)

        tree_logits = torch.concat(tree_logits_split, dim=-2)  # concat at the seq dimension
        hidden_state = torch.concat(hidden_state_splits, dim=-2)
        return tree_logits, hidden_state

    tree_pos_ids = tree_pos_ids.to(device)
    tree_mask = tree_mask.to(device)

    tree_pos_ids_split = tree_pos_ids.split(lens_split.tolist(), dim=0)
    cum_lens_split = torch.cumsum(lens_split, dim=-1)
    # [step1] end

    # [step2] start pipelined verification
    for i, subseq_len in enumerate(lens_split):
        if i == 0:
            # isend_mask.wait()
            tree_mask_split = tree_mask[..., :cum_lens_split[i], :cum_lens_split[i]]
        else:
            tree_mask_split = tree_mask[..., cum_lens_split[i-1]:cum_lens_split[i], :cum_lens_split[i]]
        
        # set the tree mask for the current stage
        stage_model.stage_base_model.model.tree_mask = tree_mask_split.contiguous()
        subseq_pos_ids = tree_pos_ids_split[i]

        if config.is_first_stage:
            subseq_ids = comm.recvfrom(src_rank=config.last_rank, device=device)
            with profiler.time_context(f"Stage {config.stage}: forward", cpu=False) if profiler is not None else nullcontext():
                outputs, sub_hidden_state = stage_model(
                    input_ids=subseq_ids,
                    past_key_values=stage_past_key_values,
                    position_ids=subseq_pos_ids,
                )
        else:
            last_hidden_state = comm.recvfrom(config.last_rank, device=device)
            with profiler.time_context(f"Stage {config.stage}: forward", cpu=False) if profiler is not None else nullcontext():
                outputs, sub_hidden_state = stage_model(
                    inputs_embeds=last_hidden_state,
                    past_key_values=stage_past_key_values,
                    position_ids=subseq_pos_ids,
                )

        comm.sendto(sub_hidden_state.cpu(), config.next_rank)
    # [step2] end

def stage_tree_decoding0(
        stage_model,
        stage_past_key_values=None,
        retrieve_indices=None,
        draft_seqs_split=None,
        tree_pos_ids_split=None,
        lens_split=None,
        input_ids=None,
        tree_mask_split=None,
):
    """
    pipelined tree decoding for verification
    :param stage_model: necessary
    :param stage_past_key_values: necessary
    :param draft_seqs_split: only necessary for stage1
    :param tree_pos_ids_split: only necessary for stage1
    :param lens_split: only necessary for stage1
    :param input_ids: only necessary for stage1
    :param retrieve_indices: only necessary for stage(-1)
    :return: return result on stage1
    """
    # [step1] prepare necessary data to all devices
    # todo: overlap with the computation of the 1st subseq on the 1st stage
    # print(f"stage {stage_model.stage} barrier")
    # dist.barrier()
    config = stage_model.config
    if stage_model.is_first_stage:
        send(retrieve_indices, dst=stage_model.total_stage-1)
    if stage_model.is_last_stage:
        hidden_states_split = []
        orig_split = []
        retrieve_indices = recv(src=0, data_type=torch.int64, shape_length=2).to(stage_model.stage_base_model.device)
        
    for i in range(stage_model.total_stage):
        if stage_model.is_first_stage:
            tree_mask = tree_mask_split[i]
            stage_model.stage_base_model.model.tree_mask = tree_mask
            position_ids = tree_pos_ids_split[i] + input_ids.shape[1]
            outputs, hidden_states = stage_model(
                input_ids=draft_seqs_split[i],
                past_key_values=stage_past_key_values,
                position_ids=position_ids
            )
            send(tree_mask, dst=1)
            send(hidden_states, dst=1)
            send(position_ids, dst=1)
            
        elif stage_model.is_last_stage:
            tree_mask = recv(src=stage_model.stage - 1, data_type=torch.float32, shape_length=4).to(stage_model.stage_base_model.device)
            stage_model.stage_base_model.model.tree_mask = tree_mask
            inputs_embeds = recv(src=stage_model.stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
            position_ids = recv(src=stage_model.stage - 1, data_type=torch.int64, shape_length=1).to(stage_model.stage_base_model.device)
            
            outputs, orig, hidden_states = stage_model(
                inputs_embeds=inputs_embeds,
                past_key_values=stage_past_key_values,
                position_ids=position_ids,
                output_orig=True,
            )
            hidden_states_split.append(hidden_states)
            orig_split.append(orig)
            
        else:
            tree_mask = recv(src=stage_model.stage - 1, data_type=torch.float32, shape_length=4).to(stage_model.stage_base_model.device)
            stage_model.stage_base_model.model.tree_mask = tree_mask
            inputs_embeds = recv(src=stage_model.stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
            position_ids = recv(src=stage_model.stage - 1, data_type=torch.int64, shape_length=1).to(stage_model.stage_base_model.device)
            
            outputs, hidden_states = stage_model(
                inputs_embeds=inputs_embeds,
                past_key_values=stage_past_key_values,
                position_ids=position_ids
            )
            send(tree_mask, dst=stage_model.stage + 1)
            send(hidden_states, dst=stage_model.stage + 1)
            send(position_ids, dst=stage_model.stage + 1)
            
    if stage_model.is_last_stage:
        orig = torch.cat(orig_split, dim=-2)
        hidden_states = torch.cat(hidden_states_split, dim=-2)
        logits = orig[0, retrieve_indices]
        return logits, hidden_states


# [MODIFIED] from update_inference_inputs()
@torch.no_grad()
def update_stage_inference_inputs(
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

    # if model.is_first_stage:
    #     retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    #     accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length]
    #     token = gen_token(prob=sample_p, logits_processor=logits_processor)[None]
        # prob = sample_p
        # new_token += accept_length
        
        # return input_ids, accept_hidden_state_new, token#, new_token


# [ADD] for continuous speculation
def token_tree_partition(draft_tokens, retrieve_indices, total_stage, subseq_len=None):
    """
    [Update] one more subseq as waiting_draft
    至少分成total_stage组(4)，如果每段长度大于subseq_len，则取subseq_len，多余的作为多出的一段(5)
    split the token tree into multiple subtrees: initial partition to start pipelined tree decoding
    :param draft_tokens:
    :param retrieve_indices:
    :param tree_position_ids:
    :param total_stage:
    :return:
    """
    # if sort in other order, process the whole input token tree here:
    # for example: dfs order (lexicographical order)
    # rows = [row.tolist() for row in retrieve_indices]
    # sorted_retrieve_indices = torch.tensor(sorted(rows))  # sort in lexicographical order
    if subseq_len is not None:
        draft_len = draft_tokens.size(-1)
        if draft_len // total_stage <= subseq_len:
            tokens_split, lens_split = split_sequence_close_equal_len(draft_tokens, total_stage)
        else:
            lens_split = [subseq_len] * total_stage + [draft_len - subseq_len * total_stage]
            tokens_split = draft_tokens.split(lens_split, dim=-1)
            lens_split = torch.tensor(lens_split, dtype=torch.long)
    else:
        tokens_split, lens_split = split_sequence_close_equal_len(draft_tokens, total_stage)
    # tree_position_ids_split = tree_position_ids.split(lens_split)

    cum_seq_lens = torch.cumsum(lens_split, dim=-1)

    ri_depth_cum = torch.zeros(retrieve_indices.size(0), dtype=torch.long)
    bottom = torch.full((retrieve_indices.size(0),), -1, dtype=torch.long)
    retrieve_indices = torch.cat((retrieve_indices, bottom[:, None]), dim=1)  # add -1 to bottom to prevent overflow
    subseq_ri_cum_depths = []

    for i, cum_seq_len in enumerate(cum_seq_lens):
        for j in range(0 if i == 0 else cum_seq_lens[i - 1], cum_seq_len):
            row_indices = torch.arange(retrieve_indices.size(0), dtype=torch.long)
            cum_ri_leaves = retrieve_indices[row_indices, ri_depth_cum]
            ri_depth_cum[cum_ri_leaves == j] += 1

        subseq_ri_cum_depths.append(ri_depth_cum.clone())

    return tokens_split, lens_split, torch.stack(subseq_ri_cum_depths, dim=0)


def fill_pipeline_stages(
    stage_model,
    stage_past_key_values=None,  # necessary for following stages
    input_ids=None,
    lens_split=None,
    draft_tokens=None,
    retrieve_indices=None,
    tree_mask=None,
    tree_pos_ids=None,
    subseq_ri_cum_depths=None,
    prof=None,
):
    config = stage_model.config
    device = stage_model.stage_base_model.device
    comm = stage_model.comm
    # print(f'Stage {config.stage} fill_pipeline_stages')


    # dist.barrier()

    # draft stage 0
    if config.is_draft_stage:  # [IMPORTANT] lens_split = world_size = 5
        cum_subseq_lens = torch.cumsum(lens_split, dim=-1)
        for i, cum_seq_len in enumerate(cum_subseq_lens):  # send 5 times
            start_ids = 0 if i == 0 else cum_subseq_lens[i-1]
            draft_tokens_split = draft_tokens[..., start_ids:cum_seq_len]
            tree_pos_ids_split = tree_pos_ids[..., start_ids:cum_seq_len]
            tree_mask_split = tree_mask[..., start_ids:cum_seq_len, :cum_seq_len]
            
            comm.send_appended(draft_tokens_split, tree_pos_ids_split, tree_mask_split.contiguous())
        return
    
    # following stages 1-4
    for i in range(config.total_stage - config.stage):
        with prof.time_context(f"Stage {config.stage}: recv from last stage", cpu=True) if prof is not None else nullcontext():
            appended_input, subseq_pos_ids, tree_mask = comm.recv_appended(device)
        # set the tree mask for the current stage
        stage_model.stage_base_model.model.tree_mask = tree_mask

        with prof.time_context(f"Stage {config.stage}: draft init forward", cpu=False) if prof is not None else nullcontext():
            if config.is_first_stage:            
                outputs, sub_hidden_state = stage_model(
                    input_ids=appended_input,
                    past_key_values=stage_past_key_values,
                    position_ids=subseq_pos_ids,
                )
            else:
                outputs, sub_hidden_state = stage_model(
                    inputs_embeds=appended_input,
                    past_key_values=stage_past_key_values,
                    position_ids=subseq_pos_ids,
                )

        if config.is_last_stage:
            comm.sendto(sub_hidden_state, config.next_rank)
        else:
            comm.send_appended(sub_hidden_state, subseq_pos_ids, tree_mask)        

# def fill_pipeline_stages(
#         stage_model,
#         stage_past_key_values=None,
#         input_ids=None,
#         lens_split=None,
#         draft_tokens=None,
#         retrieve_indices=None,
#         tree_mask=None,
#         tree_pos_ids=None,
#         subseq_ri_cum_depths=None,
# ):
#     # print('============fill_pipeline_stages============')
#     config = stage_model.config
#     device = stage_model.stage_base_model.device
#     comm = stage_model.comm  # update

#     # [update] draft stage async broadcast tree_info
#     # 这里的广播其实是同步的？
#     if config.is_draft_stage:
#         broadcast_tree_info_task = comm.executor.submit(
#             comm.broadcast_tree_info,
#             lens_split,
#             tree_pos_ids,
#             tree_mask,
#             draft_tokens,
#             retrieve_indices,
#             subseq_ri_cum_depths,
#             appended=False
#         )
#         draft_tokens_split = draft_tokens.split(lens_split.tolist(), dim=-1)
#         for split_draft_tokens in draft_tokens_split:
#             comm.sendto(split_draft_tokens, dst_rank=config.next_rank)
            
#         broadcast_tree_info_task.result()
#         # end for draft stage
#         return
#     else:
#         if not config.is_last_stage:
#             lens_split, tree_pos_ids, tree_mask = comm.broadcast_tree_info_global(appended=False)
#         else:
#             lens_split, tree_pos_ids, tree_mask, draft_tokens, retrieve_indices, subseq_ri_cum_depths = comm.broadcast_tree_info(appended=False)
#         tree_mask = tree_mask.to(device)
#         tree_pos_ids = tree_pos_ids.to(device)

#         cum_lens_split = torch.cumsum(lens_split, dim=-1)
#         tree_pos_ids_split = tree_pos_ids.split(lens_split.tolist(), dim=0)
    
#     # fill the pipeline stages
#     # stage1 doesn't need to recv anything
#     for i in range(config.total_stage - config.stage):
#         # print(config.stage, f'i={i}')
#         if i == 0:
#             # isend_mask.wait()
#             tree_mask_split = tree_mask[..., :cum_lens_split[i], :cum_lens_split[i]]
#         else:
#             tree_mask_split = tree_mask[..., cum_lens_split[i-1]:cum_lens_split[i], :cum_lens_split[i]]
        
#         # set the tree mask for the current stage
#         stage_model.stage_base_model.model.tree_mask = tree_mask_split.contiguous()
#         subseq_pos_ids = tree_pos_ids_split[i]

#         if config.is_first_stage:
#             subseq_ids = comm.recvfrom(src_rank=config.last_rank, device=device)
            
#             outputs, sub_hidden_state = stage_model(
#                 input_ids=subseq_ids,
#                 past_key_values=stage_past_key_values,
#                 position_ids=subseq_pos_ids,
#             )
#         else:
#             last_hidden_state = comm.recvfrom(config.last_rank, device=device)
#             outputs, sub_hidden_state = stage_model(
#                 inputs_embeds=last_hidden_state,
#                 past_key_values=stage_past_key_values,
#                 position_ids=subseq_pos_ids,
#             )
#         # if i == 0:
#         #     if config.is_first_stage:
#         #         broadcast_tree_info_task.result()

#         if i < config.total_stage - config.stage - 1:  # not send for the last time
#             comm.sendto(sub_hidden_state.cpu(), config.next_rank)
        
#     # if config.is_first_stage:
#     #     return sub_hidden_state
#     if config.is_last_stage:
#         return sub_hidden_state, lens_split, draft_tokens, retrieve_indices, tree_mask, tree_pos_ids, subseq_ri_cum_depths
#     # middle stages
#     return sub_hidden_state, lens_split, tree_mask, tree_pos_ids
    

# generated by Qwen2.5Max
def get_subtree_retrieve_indices(retrieve_indices, cum_depth):
    """
    Get the retrieve_indices of a subtree according to the cumulate depth of each path
    :param retrieve_indices: (path * max_tree_depth)
    :param cum_depth: (path,) < max_tree_depth each
    :return:
    """
    paths, depth = retrieve_indices.shape
    max_cum_depth = cum_depth.max().item()

    # mark the padding
    mask = torch.arange(max_cum_depth).unsqueeze(0) < cum_depth.unsqueeze(1)  # (paths, max_cum_depth)
    # initialize with "-1"
    result = torch.full((paths, max_cum_depth), -1, dtype=retrieve_indices.dtype)
    # fill with data
    result[mask] = retrieve_indices[:, :max_cum_depth][mask[:, :depth]]
    return result


def find_prefix_match(retrieve_indices, accept_indices):
    prefixes = retrieve_indices[:, :accept_indices.size(0)]

    matches = torch.all(prefixes == accept_indices.unsqueeze(0), dim=1)

    match_paths = torch.nonzero(matches).squeeze(1)

    return match_paths


def process_retrieve_indices(retrieve_indices):
    flattened = retrieve_indices.reshape(-1)
    mask = flattened != -1
    filtered = flattened[mask]

    unique_values = torch.unique(filtered)
    sorted_values = torch.sort(unique_values).values

    return sorted_values


def map_retrieve_indices(retrieve_indices, a, b):
    # consider a is sorted, transform elements in retrieve_indices by mapping a->b
    assert a.size(0) == b.size(0), f'a.size(0)={a.size(0)}, b.size(0)={b.size(0)}'
    flat = retrieve_indices.reshape(-1)
    mask = flat != -1
    if not mask.any():
        return torch.full_like(retrieve_indices, -1)
    indices = torch.searchsorted(a, flat[mask])
    valid_mask = indices < len(a)
    result = torch.full_like(flat, -1)
    result[mask] = b[indices[valid_mask]]
    return result.view(retrieve_indices.shape)


def cal_pruning_info(draft_tokens, retrieve_indices, best_candidate, accept_len, new_token, subseq_ri_cum_depths):
    """
    Prune the token tree on the retrieve_indices
    :param retrieve_indices:
    :param best_candidate:
    :param accept_len:
    :return:
    """
    # accept_len > 0
    accepted_indices = retrieve_indices[best_candidate, :accept_len]

    # judge whether the global leaf node is reached
    cur_path_depth = (retrieve_indices[best_candidate, :] != -1).sum().item()
    if accept_len == retrieve_indices.size(-1) or retrieve_indices[best_candidate, accept_len] == -1:  # the next token of the accepted
        # truncate: reach the global leaf: occurs in the eagenerate_pruned_pipeline()
        # print('- leaf has been reached')
        return accepted_indices, True

    # judge whether the new token follows the tree
    matched_candidates = find_prefix_match(retrieve_indices, accepted_indices)  # non-zero
    next_indices_draft = retrieve_indices[matched_candidates, accept_len]  # todo: retrieve_indices会比last stage的draft tokens多，每次appended只会把append的token给到last stage
    next_tokens_draft = draft_tokens[0, next_indices_draft]
    # found the paths with prefix of "accept_tokens + new_token"

    same_indices = torch.nonzero(next_tokens_draft.cpu() == new_token.cpu()).squeeze(1)
    if same_indices.numel() == 0:
        # truncate: unmatched token
        # print(f'- - no match token found in the tree, accept_len/global_depth={accept_len}/{cur_path_depth}')
        # print(f'- - next_tokens_draft={next_tokens_draft.unique()}')
        # print(f'- - new_token={new_token}')
        return accepted_indices, True

    # pruning
    left_candidates = matched_candidates[same_indices]
    left_retrieve_indices = retrieve_indices[left_candidates, accept_len:]  # todo: left_retrieve_indices all -1

    # [update]: retrieve_indices may be larger than draft_tokens
    # left_indices_global: for the global context
    # left_indices: for the tree in pipeline
    max_depth = (left_retrieve_indices != -1).sum(dim=1).max().item()
    left_retrieve_indices = left_retrieve_indices[:, :max_depth]  # drop the all -1 parts
    left_indices_global = process_retrieve_indices(left_retrieve_indices)  # for pruning for other stages
    left_indices_global = torch.cat((accepted_indices, left_indices_global), dim=0)
    left_indices_from_zero = torch.arange(left_indices_global.size(-1) - accept_len, dtype=torch.long)

    left_indices = left_indices_global[left_indices_global < draft_tokens.size(-1)]

    return left_indices, False


# [update] draft stage pruning
def draft_stage_pruning(
    left_indices,
    accept_len,
    draft_tokens,
    tree_mask,
    tree_pos_ids,
    retrieve_indices,
    subseq_ri_cum_depths=None,
    lens_split=None,
    ):
    prefix_indices = left_indices[:accept_len+1]  # including the new token
    # print(f'stage0 prefix_indices={prefix_indices}, draft_tokens={draft_tokens.shape}')
    accepted_tokens = draft_tokens[:, left_indices[:accept_len]]

    # prune retrieve_indices
    matched_candidates = find_prefix_match(retrieve_indices, prefix_indices)
    left_retrieve_indices = retrieve_indices[matched_candidates, accept_len:]
    draft_stage_left_indices = process_retrieve_indices(left_retrieve_indices)

    # # prune draft_tokens
    left_draft_tokens = draft_tokens[:, draft_stage_left_indices]

    # print(f'stage0 left_retrieve_indices:{left_retrieve_indices}')
    max_depth = (left_retrieve_indices != -1).sum(dim=1).max().item()
    left_retrieve_indices = left_retrieve_indices[:, :max_depth]  # drop the all -1 layers
    left_indices_from_zero = torch.arange(draft_stage_left_indices.size(-1), dtype=torch.long)
    transformed_ri = map_retrieve_indices(left_retrieve_indices, draft_stage_left_indices, left_indices_from_zero)

    draft_stage_left_indices = torch.cat((prefix_indices[:-1], draft_stage_left_indices), dim=0)
    
    # prune tree_mask
    if tree_mask is not None:
        tree_mask_cpu = tree_mask.cpu()
        tree_mask_left_indices = left_indices[accept_len:]
        # assert torch.max(tree_mask_left_indices) < tree_mask_cpu.size(-1), f'stage{stage} tree_mask_left_indices={tree_mask_left_indices} is out of range'
        tree_mask = tree_mask_cpu[..., tree_mask_left_indices[:, None], tree_mask_left_indices].to(tree_mask.device)

    # prune tree_pos_ids
    if tree_pos_ids is not None:
        tree_pos_ids_cpu = tree_pos_ids.cpu()
        tree_pos_ids = tree_pos_ids_cpu[tree_mask_left_indices].to(tree_pos_ids.device)
        
    try:
        assert left_draft_tokens.size(-1) + accept_len == draft_stage_left_indices.size(-1)
    except AssertionError as e:
        print(f'left_draft_tokens + accept_len != draft_stage_left_indices: {left_draft_tokens.size(-1)} and {draft_stage_left_indices.size(-1)}')
        print(f'left_indices: {left_indices}')
        print(f'draft_stage_left_indices: {draft_stage_left_indices}')
        print(f'accept_len: {accept_len}')
        print(f'prefix_indices: {prefix_indices.size(-1)}')
        assert draft_stage_left_indices.max() < draft_tokens.size(-1), f'draft_stage_left_indices.max()={draft_stage_left_indices.max()} is out of range'
        raise e

    # prune subseq_ri_cum_depths
    if subseq_ri_cum_depths is not None:  # continuous
        left_ri_cum_depths = subseq_ri_cum_depths[1:, matched_candidates] - accept_len
        cum_lens = torch.cumsum(lens_split, dim=0)
        lens_split = torch.tensor([torch.sum((left_indices >= cum_lens[i-1]) & (left_indices < cum_lens[i])) for i in range(1, cum_lens.size(-1))], dtype=torch.long)
        return left_draft_tokens, tree_mask, tree_pos_ids, transformed_ri, accepted_tokens, left_ri_cum_depths, draft_stage_left_indices, lens_split

    # pruned-only
    return left_draft_tokens, tree_mask, tree_pos_ids, transformed_ri, accepted_tokens

def last_hs_pruning(
    hidden_state,
    left_indices,
    global_accept_len,
    accept_len,
):
    if hidden_state is not None:
        cur_hs_len = hidden_state.size(-2)

        hs_indices_start_idx = cur_kv_len - cur_hs_len  # [update]: idx should minus 1 additionally
        left_indices_in_hs = left_indices_in_cache[(left_indices_in_cache >= hs_indices_start_idx) & (left_indices_in_cache < cur_kv_len)] - hs_indices_start_idx
        left_indices_in_hs = left_indices_in_hs.to(hidden_state.device)

        if left_indices_in_hs.numel() > 0:
            assert torch.max(left_indices_in_hs) < hidden_state.size(-2), f'stage{stage} left_indices_in_hs={left_indices_in_hs} is out of range'
        hidden_state = hidden_state[..., left_indices_in_hs, :]  # todo: bug occurs when hidden_state is empty tensor    
    

def token_pruning(
        past_key_values_data_list,
        current_length_data,
        lens_split,
        last_hidden_state,
        tree_mask,
        tree_pos_ids,
        left_indices,
        global_accept_len,
        accept_len,
        stage
):
    """
    prune the tokens related data: kv-cache, last_hidden_state, tree_mask
    [update] last_hidden_state is the output of last_rank
    """
    cur_kv_len = current_length_data[0].item()
    cache_device = past_key_values_data_list[0][0].device

    # prune cache
    # ***这里的global_accept_len还是没有加上accept_len长度的
    left_indices_global = left_indices + global_accept_len
    left_indices_in_cache = (left_indices_global[left_indices_global < cur_kv_len])
    left_indices_after_cache = left_indices_global[left_indices_in_cache.size(-1):]
    
    # copy and set cache length
    left_indices_in_cache_size = left_indices_in_cache.size(-1)
    for past_key_values_data in past_key_values_data_list:
        left_kv_cache = past_key_values_data[..., left_indices_in_cache, :]
        cache_dst = past_key_values_data[..., global_accept_len:global_accept_len+left_indices_in_cache_size, :]
        cache_dst.copy_(left_kv_cache, non_blocking=True)
    current_length_data.fill_(global_accept_len + left_indices_in_cache_size)

    # prune lens_split
    if lens_split is not None:
        # test kv_len
        assert cur_kv_len == global_accept_len + torch.sum(lens_split[:5-stage]), f'stage{stage} wrong kv_len={cur_kv_len} while global_accept={global_accept_len} and lens_split={lens_split}'

        cum_lens = torch.cumsum(lens_split, dim=0)
        lens_split = torch.tensor([torch.sum((left_indices >= cum_lens[i-1]) & (left_indices < cum_lens[i])) for i in range(1, cum_lens.size(-1))], dtype=torch.long)

    # prune last_hidden_state
    # the last_hidden_state here are the output of the current stage_model
    if last_hidden_state is not None:
        cur_hs_len = last_hidden_state.size(1) 

        hs_indices_start_idx = cur_kv_len  # [update]: last_hidden_state is new to current stage
        hs_indices_end_idx = cur_kv_len + cur_hs_len
        left_indices_in_input = left_indices_after_cache[left_indices_after_cache < hs_indices_end_idx] - hs_indices_start_idx  # prune last_hidden_state, tree_mask, tree_pos_ids
        # left_indices_in_hs = left_indices_in_hs.to(last_hidden_state.device)
        if left_indices_in_input.numel() > 0:
            assert tree_pos_ids.size(0) == tree_mask.size(2) > max(left_indices_in_input), f'last_hidden_state.shape={last_hidden_state} tree_pos_ids.shape={tree_pos_ids.shape}, tree_mask.shape={tree_mask.shape}, left_indices_in_input={left_indices_in_input}'

        if left_indices_in_input.numel() > 0:
            assert torch.max(left_indices_in_input) < last_hidden_state.size(1), f'stage{stage} left_indices_in_input={left_indices_in_input} is out of range'
        if len(last_hidden_state.shape) == 3:
            last_hidden_state = last_hidden_state[..., left_indices_in_input.to(last_hidden_state.device), :]  # todo: bug occurs when hidden_state is empty tensor
        else:
            last_hidden_state = last_hidden_state[..., left_indices_in_input.to(last_hidden_state.device)]  # todo: bug occurs when hidden_state is empty tensor
        
    # prune tree_mask
    if tree_mask is not None:
        tree_mask_cpu = tree_mask.cpu()
        local_tree_mask_left_indices = left_indices_in_input
        global_tree_mask_left_indices = left_indices[accept_len:]
        global_tree_mask_left_indices = global_tree_mask_left_indices[global_tree_mask_left_indices < tree_mask_cpu.size(-1)]
        # assert torch.max(tree_mask_left_indices) < tree_mask_cpu.size(-1), f'stage{stage} tree_mask_left_indices={tree_mask_left_indices} is out of range'
        tree_mask = tree_mask_cpu[..., local_tree_mask_left_indices[:, None], global_tree_mask_left_indices].to(tree_mask.device)

    # prune tree_pos_ids
    if tree_pos_ids is not None:
        tree_pos_ids_cpu = tree_pos_ids.cpu()
        
        tree_pos_ids = tree_pos_ids_cpu[local_tree_mask_left_indices].to(tree_pos_ids.device)

    return past_key_values_data_list, current_length_data, last_hidden_state, tree_mask, tree_pos_ids

def get_parent_indices_np(tree_mask):
    """
    Compute parent indices for each node in a tree using NumPy.
    
    Args:
        tree_mask (np.ndarray): Shape [n, n], boolean matrix indicating parent-child relationships.

    Returns:
        np.ndarray: Shape [n], where parent_indices[i] = j means j is the last parent of i, or -1 if no parent.
    """
    n = tree_mask.shape[0]
    
    tree_mask = tree_mask.astype(np.bool_)
    offset_mask = np.tri(n, n, k=-1, dtype=np.bool_)
    masked = np.logical_and(tree_mask, offset_mask)
    flipped_masked = np.fliplr(masked)
    col_indices = np.argmax(flipped_masked, axis=1)
    parent_indices = n - 1 - col_indices
    all_false_rows = ~flipped_masked[np.arange(n), col_indices]
    parent_indices[all_false_rows] = -1

    return parent_indices
    
def merge_two_tree(
        tree1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tree2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        lens_split,
        subseq_ri_cum_depths,
        prof=None
):
    """
    Merge two tree that share the same root node, tree1 is the old tree, tree2 is the new tree
    The merged tree has the draft_tokens of: {draft_tokens1, added_tokens}
    Ensure that **the whole tree is on CPU**
    """
    with prof.time_context(f"init", cpu=True) if prof is not None else nullcontext():
        draft_tokens1, retrieve_indices1, tree_mask1, tree_pos_ids1 = tree1
        draft_tokens2, retrieve_indices2, tree_mask2, tree_pos_ids2 = tree2

        tree1_depth = retrieve_indices1.size(1)
        tree2_depth = retrieve_indices2.size(1)
        tree_mask1 = tree_mask1[0, 0, ...]
        tree_mask2 = tree_mask2[0, 0, ...]
        tree1_size = draft_tokens1.size(-1)

        tree_mask1_np = tree_mask1.numpy()
        tree_mask2_np = tree_mask2.numpy()
        draft_tokens1_np = draft_tokens1[0].numpy()
        draft_tokens2_np = draft_tokens2[0].numpy()
        retrieve_indices1_np = retrieve_indices1.numpy()
        retrieve_indices2_np = retrieve_indices2.numpy()

    # get paths of draft_tokens1
    with prof.time_context(f"get paths of draft_tokens1", cpu=True) if prof is not None else nullcontext():
        tree_mask1_np = tree_mask1.numpy()
        tree1_token_paths_idx = [(tuple(draft_tokens1_np[np.flatnonzero(tree_mask1_np[i])]), i) for i in range(tree_mask1_np.shape[0])]
        paths_tree1_idx = dict(tree1_token_paths_idx)

    # go through draft_tokens2
    with prof.time_context(f"go through draft_tokens2", cpu=True) if prof is not None else nullcontext():
        # paths_tree2 = set()
        index_mapping_2_to_merged = np.zeros(draft_tokens2.size(1), dtype=np.int64)
        append_indices = []
        tree2_token_paths = [tuple(draft_tokens2_np[np.flatnonzero(tree_mask2_np[i])]) for i in range(tree_mask2_np.shape[0])]
        paths_tree2 = set(tree2_token_paths)

        for i, token_path in enumerate(tree2_token_paths):
            if len(token_path) <= tree1_depth and token_path in paths_tree1_idx:
                index_mapping_2_to_merged[i] = paths_tree1_idx[token_path]
            else:
                mapped_idx = tree1_size + len(append_indices)
                append_indices.append(i)
                index_mapping_2_to_merged[i] = mapped_idx

    with prof.time_context(f"merge tokens and positions", cpu=True) if prof is not None else nullcontext():
        append_indices = torch.tensor(append_indices, dtype=torch.long)
        draft_tokens_merged = torch.cat((draft_tokens1, draft_tokens2[:, append_indices]), dim=1)
        merged_tree_pos_ids = torch.cat((tree_pos_ids1, tree_pos_ids2[append_indices]), dim=0)
    assert draft_tokens_merged.size(-1) == merged_tree_pos_ids.size(0), f'draft_tokens_merged != merged_tree_pos_ids: {draft_tokens_merged.size(-1)} and {merged_tree_pos_ids.size(0)}'
    # [merge tree_mask]
    with prof.time_context(f"merge tree_mask", cpu=True) if prof is not None else nullcontext():
        merged_size = draft_tokens_merged.size(-1)
        # init merged_tree_mask as tree_mask1
        merged_tree_mask = np.zeros((merged_size, merged_size), dtype=tree_mask1_np.dtype)
        merged_tree_mask[:tree1_size, :tree1_size] = tree_mask1_np

        with prof.time_context(f"get parent indices", cpu=True) if prof is not None else nullcontext():
            parent_indices = get_parent_indices_np(tree_mask2_np)
        with prof.time_context(f"iterative merge tree_mask", cpu=True) if prof is not None else nullcontext():
            for i, append_idx in enumerate(append_indices):
                mapped_idx = index_mapping_2_to_merged[append_idx]
                parent_idx = index_mapping_2_to_merged[parent_indices[append_idx]]
                mapped_parent_mask_row = merged_tree_mask[parent_idx, :parent_idx+1]
                merged_tree_mask[mapped_idx, :parent_idx+1] = mapped_parent_mask_row
                merged_tree_mask[mapped_idx, mapped_idx] = 1

    with prof.time_context(f"merge retrieve_indices", cpu=True) if prof is not None else nullcontext():
        leaf_depths1 = (retrieve_indices1_np != -1).sum(axis=1)
        leaf_depths2 = (retrieve_indices2_np != -1).sum(axis=1)
        leave_paths1 = [(tuple(draft_tokens1_np[retrieve_indices1_np[i, :leaf_depths1[i]]]), i) for i in range(retrieve_indices1_np.shape[0])]
        leave_paths1 = dict(leave_paths1)
        leave_paths2 = [(tuple(draft_tokens2_np[retrieve_indices2_np[i, :leaf_depths2[i]]]), i) for i in range(retrieve_indices2_np.shape[0])]
        leave_paths2 = dict(leave_paths2)

        selected_leaves1 = np.zeros(retrieve_indices1.size(0), dtype=np.bool_)
        selected_leaves2 = np.zeros(retrieve_indices2.size(0), dtype=np.bool_)
        for leaf_path, leaf_path_idx in leave_paths1.items():
            if leaf_path in paths_tree2 and leaf_path not in leave_paths2:
                pass
            else:
                selected_leaves1[leaf_path_idx] = True
        
        for leaf_path, leaf_path_idx in leave_paths2.items():
            if leaf_path not in paths_tree1_idx:
                selected_leaves2[leaf_path_idx] = True

        tree1_selected_sum = selected_leaves1.sum()
        tree2_selected_sum = selected_leaves2.sum()
        merged_selected_sum = tree1_selected_sum + tree2_selected_sum
        max_depth = max(tree1_depth, tree2_depth)
        ri_merged = np.full((merged_selected_sum, max_depth), -1, dtype=np.int64)
        ri_merged[:tree1_selected_sum, :tree1_depth] = retrieve_indices1_np[selected_leaves1]
        ri2_merged = retrieve_indices2_np[selected_leaves2]
        valid_mask = ri2_merged != -1
        ri2_merged[valid_mask] = index_mapping_2_to_merged[ri2_merged[valid_mask]]
        
        ri_merged[tree1_selected_sum:merged_selected_sum, :tree2_depth] = ri2_merged
        retrieve_indices_merged = torch.from_numpy(ri_merged)
    # [merge retrieve_indices] finish

    # update lens_split and subseq_ri_cum_depths
    with prof.time_context(f"update lens_split and subseq_ri_cum_depths", cpu=True) if prof is not None else nullcontext():
        lens_split = torch.cat((lens_split, torch.tensor([append_indices.size(0)], dtype=torch.long)))
        # todo: len_split多长？subseq_ri_cum_depths应该多长？
        n_leaves = retrieve_indices_merged.size(0)
        subseq_ri_cum_depths = []
        cum_seq_lens = np.cumsum(lens_split[:-1].numpy(), axis=0)
        bottom = np.full((n_leaves, 1), -1, dtype=np.int64)
        retrieve_indices_filled = np.concatenate((retrieve_indices_merged.numpy(), bottom), axis=1)  # add -1 to bottom to prevent overflow

        ri_depth_cum = np.zeros(n_leaves, dtype=np.int64)
        for i, cum_seq_len in enumerate(cum_seq_lens):
            for j in range(0 if i == 0 else cum_seq_lens[i - 1], cum_seq_len):
                row_indices = np.arange(n_leaves, dtype=np.int64)
                cum_ri_leaves = retrieve_indices_filled[row_indices, ri_depth_cum]
                ri_depth_cum[cum_ri_leaves == j] += 1
            # update: 只计算到在pipeline里的draft token tree部分，即将输入的最新一段单独算
            subseq_ri_cum_depths.append(ri_depth_cum.copy())
        subseq_ri_cum_depths = np.stack(subseq_ri_cum_depths, axis=0)
    
    return draft_tokens_merged, retrieve_indices_merged, torch.from_numpy(merged_tree_mask)[None, None], merged_tree_pos_ids, lens_split, torch.from_numpy(subseq_ri_cum_depths)


def expand_tree(
    draft_tokens,
    retrieve_indices,
    tree_mask,
    tree_pos_ids,
    lens_split,
    subseq_ri_cum_depths
):
    with prof.profile_context(f"Stage {config.stage}: topK_genrate", device=f"cuda:{device}") if prof is not None else nullcontext():
        draft_tokens2, retrieve_indices2, tree_mask2, tree_position_ids2 = self.ea_layer.topK_genrate(
            accepted_hidden_state,
            input_ids_ea,
            self.stage_base_model.lm_head,
            logits_processor,
            depth=max(cur_draft_depth + 2, 5)  # todo: test best tree settings
            # total_tokens=64
        )  # get a little more appended tokens
    tree_position_ids2 = tree_position_ids2 + input_ids.size(-1)

    # print(tree_position_ids[0], tree_position_ids2[0])

    assert draft_tokens.size(-1) == tree_position_ids.size(0), f'draft_tokens != tree_pos_ids: {draft_tokens.size(-1)} and {tree_position_ids.size(0)}'

    with prof.time_context(f"Stage {config.stage}: merge_two_tree", cpu=True) if prof is not None else nullcontext():
        # [update] operate on CPU
        origin_device = draft_tokens.device
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, lens_split, subseq_ri_cum_depths = merge_two_tree(
            (draft_tokens.cpu(), retrieve_indices.cpu(), tree_mask.cpu(), tree_position_ids.cpu()),
            (draft_tokens2.cpu(), retrieve_indices2.cpu(), tree_mask2.cpu(), tree_position_ids2.cpu()),
            lens_split,
            subseq_ri_cum_depths
        )
        draft_tokens = draft_tokens.to(origin_device)
        tree_mask = tree_mask.to(origin_device)
        tree_position_ids = tree_position_ids.to(origin_device)
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, lens_split, subseq_ri_cum_depths


# EAGLE
def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max().item()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]  # 直接跳过树的根节点
        best_candidate = 0
        if candidates.size(-1) == 1:
            # print('stage3: Only one token, accept it')
            # print(f'evaluate_posterior: accept_cand={accept_cand}')
            gt_logits = logits[best_candidate, 0][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            return torch.tensor(best_candidate), accept_length - 1, torch.softmax(gt_logits, dim=0)

        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            # print('--  reach leaf')
            gt_logits = logits[best_candidate, accept_length - 1]
            gt_logits = logits_processor(None, gt_logits)
            sample_p = torch.softmax(gt_logits, dim=0)
        # print(f'evaluate_posterior: accept_cand={accept_cand}')
        return torch.tensor(best_candidate), accept_length - 1, sample_p


if __name__ == '__main__':
    sequence = torch.randint(0, 10000, (1, 209), dtype=torch.long)
    res = split_sequence_close_equal_len(sequence, 6)
    for i in res:
        print(i)
