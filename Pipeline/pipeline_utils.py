# reference: https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/utils.py - def evaluate_posterior()
# codes with comment "EAGLE" are original EAGLE codes

import random
import copy
import time
import torch
import torch.nn.functional as F
from queue import deque
from typing import Union, Iterable, List
import torch.distributed as dist
from memory_profiler import profile
# from stage_ea_model import StageEaModel
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from tools.communicator import*
from typing import Tuple
TOPK = 10  # topk for sparse tree

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
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


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
        return [base_size if i >= reminder else base_size + 1 for i in range(n)]


def split_sequence_close_equal_len(sequence: torch.Tensor, split_cnt: Union[int, Iterable[int], list]): #  tuple is for tree partition
    if len(sequence.shape) <= 2:
        seq_len = sequence.size(-1)
    else:
        raise ValueError('Sequence for splitting can not have a batch_size larger than 2.')

    if isinstance(split_cnt, int):
        split_lens = split_close_equal(seq_len, split_cnt)
        # print(f"split_lens={split_lens}")
    else:
        split_lens = split_cnt
    # split_lens = torch.tensor(split_lens, dtype=torch.int32)

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
        token = torch.argmax(prob)
        token = token[None, None]

    return token


def pipeline_prefill(
        stage_model,
        input_ids=None,
        stage_past_key_values=None,
        # logits_processor=None,
        # tree_mask=None
):
    config = stage_model.config
    stage_base_model = stage_model.stage_base_model
    device = stage_base_model.device
    total_stage = config.total_stage
    if config.is_first_stage:
        if input_ids.size(-1) > 50:  # pipelined prefill, split the input_ids when long enough
            seq_splits, lens_split = split_sequence_close_equal_len(input_ids, total_stage)
            hidden_state_splits = []  # save the hidden_state for later transmission
        else:
            lens_split = torch.zeros(total_stage, dtype=torch.long)
            lens_split[0] = input_ids.size(-1)
            seq_splits = (input_ids,)
    else:
        lens_split = torch.zeros(total_stage, dtype=torch.long)
    dist.broadcast(lens_split, src=0)
    lens_split = lens_split[lens_split > 0]
    isend_task = None

    for i in range(lens_split.size(-1)):
        if config.is_first_stage:
            outputs, sub_hidden_state = stage_model(
                input_ids=seq_splits[i],
                past_key_values=stage_past_key_values,
            )
            sub_hidden_state = sub_hidden_state.cpu()
            hidden_state_splits.append(sub_hidden_state)
            if isend_task is not None:
                isend_task.wait()

            isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)

        else:
            # print((1, lens_split[i], config.hidden_size))
            last_hidden_state = torch.zeros((1, lens_split[i], config.hidden_size), dtype=torch.float16)
            dist.recv(last_hidden_state, src=config.last_rank)
            last_hidden_state = last_hidden_state.to(device)

            # middle stage
            outputs = stage_base_model.model(
                inputs_embeds=last_hidden_state,
                past_key_values=stage_past_key_values,
            )
            sub_hidden_state = outputs[0]
            if isend_task is not None:
                isend_task.wait()
            sub_hidden_state = sub_hidden_state.cpu()
            isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)
    
    # print(f'stage{config.stage} wait the last send task in pipeline prefill')
    if isend_task is not None:
        isend_task.wait()
    # print(f'stage{config.stage} wait done')

    if config.is_first_stage:
        orig = ()

        for i, hidden_state_split in enumerate(hidden_state_splits):
            dist.recv(hidden_state_split, src=config.last_rank)
            hidden_state_splits[i] = hidden_state_split.to(device)
            orig = orig + (stage_base_model.lm_head(hidden_state_splits[i]),)

        hidden_state = torch.concat(hidden_state_splits, dim=-2)
        orig = torch.concat(orig, dim=-2)

        return orig, hidden_state
    

def prefill_pipeline(stage_model, stage_past_key_values=None, input_ids = None):
    # print(f"stage_model.stage: {stage_model.stage}, layer_range: {stage_model.stage_base_model.model.config.layer_range}")
    if stage_model.is_first_stage:
        outputs, hidden_states = stage_model(
            input_ids=input_ids,
            past_key_values=stage_past_key_values,
        )
        # print(f"stage {stage_model.stage} hidden_states shape: {hidden_states.shape} hidden_states dtype: {hidden_states.dtype} sending to stage {stage_model.stage + 1}")
        send(hidden_states, dst=1)
    else:
        # print(f"stage {stage_model.stage} is waiting for input_embeds from stage {stage_model.stage - 1}")
        # print(f"stage {stage_model.stage} stage base model device: {stage_model.stage_base_model.device}")
        inputs_embeds = recv(src=stage_model.stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
        # print(f"stage {stage_model.stage} input_embeds shape: {inputs_embeds.shape}")
        if stage_model.is_last_stage:
            outputs, orig, hidden_states = stage_model(
                inputs_embeds=inputs_embeds,
                past_key_values=stage_past_key_values,
                output_orig=True,
            )
            # print(f"stage {stage_model.stage} orig shape: {orig.shape}")
            # print(f"stage {stage_model.stage} hidden_states shape: {hidden_states.shape}")
            send(orig, dst=0)
            send(hidden_states, dst=0)
        else:
            outputs, hidden_states = stage_model(
                inputs_embeds=inputs_embeds,
                past_key_values=stage_past_key_values,
            )
            # print(f"stage {stage_model.stage} hidden_states shape: {hidden_states.shape} sending to stage {stage_model.stage + 1}")
            send(hidden_states, dst=stage_model.stage + 1)
            
    if stage_model.is_first_stage:
        orig = recv(src=stage_model.total_stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
        hidden_states = recv(src=stage_model.total_stage - 1, data_type=torch.float16, shape_length=3).to(stage_model.stage_base_model.device)
        # print(f"orig: {orig}")
        # print(f"hidden_states: {hidden_states}")
        return orig, hidden_states

# [MODIFIED] from initialize_tree()
def initialize_tree_pipeline(stage_model, past_key_values, logits_processor = None, input_ids = None):
    if stage_model.is_first_stage:
        # orig, hidden_states = prefill_pipeline(stage_model, past_key_values, input_ids)
        orig, hidden_states = pipeline_prefill(stage_model, input_ids, past_key_values)

        
        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(None, logits)
            # print(f"stage {stage_model.stage} logits shape: {logits.shape} logits dtype: {logits.dtype}")
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            # print(f"probabilities: {probabilities}")
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(orig[:, -1])
            token = token[None, None]
        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        # print(f"prefill token: {token}")
        # Clone the output hidden states
        
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = stage_model.ea_layer.topK_genrate(
            hidden_states,
            input_ids,
            stage_model.stage_base_model.lm_head,
            logits_processor
        )
        # print(token, draft_tokens[0, 0:1])
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token
    else:
        pipeline_prefill(stage_model, stage_past_key_values=past_key_values)

# # [ADD] for pipeline
# def tree_partition_pipeline(draft_tokens, total_stage: int):
#     """
#     split the input sequence into multiple subsequences
#     :param total_stage:
#     :param draft_tokens: flattened tree token sequence following the causal rule
#     :param tree_position_ids:
#     :return: Tuple(subtree1=(draft_tokens, tree_position_ids, tree_attention_mask, retrieve_indices),...)
#     """
#     seqs_split, lens_split = split_sequence_close_equal_len(draft_tokens, total_stage)
#     # tree_pos_ids_split = tree_position_ids.split(tuple(lens_split))
#     # print(f"lens_split={lens_split}")
#     # cum_seq_lens = torch.cumsum(lens_split, dim=-1)
    
#     # # split tree mask for seqs_split respectively
#     # tree_mask_split = []
#     # for i in range(len(lens_split)):
#     #     if i > 0: # make sure the tree mask is contiguous, could be optimized
#     #         tree_mask_split.append(tree_mask[..., cum_seq_lens[i-1]:cum_seq_lens[i], :cum_seq_lens[i]].contiguous())
#     #     else:
#     #         tree_mask_split.append(tree_mask[..., :cum_seq_lens[i], :cum_seq_lens[i]].contiguous())
    
#     # return seqs_split, tree_pos_ids_split, tree_mask_split, lens_split
#     return seqs_split, lens_split


# [MODIFIED] from tree_decoding()
def stage_tree_decoding_liux(
        stage_model,
        stage_past_key_values=None,
        draft_seqs_split=None,
        lens_split=None,
        tree_pos_ids=None,
        tree_mask=None,
        input_ids=None,
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
    config = stage_model.config
    device = stage_model.stage_base_model.device
    if stage_model.is_first_stage:
        # send mask and pos_ids
        dist.broadcast(lens_split, src=0)
        tree_pos_ids = tree_pos_ids + input_ids.size(-1)
        dist.broadcast(tree_pos_ids, src=0)
        isend_mask = dist.broadcast(tree_mask, src=0, async_op=True)
        # dist.broadcast(tree_mask, src=0)
        # print(f'stage {stage_model.stage} tree_mask: {tree_mask.shape}, dtype: {tree_mask.dtype}')
        tree_logits_split = ()
        hidden_state_splits = []

    else:
        # recv mask
        lens_split = torch.zeros(stage_model.total_stage, dtype=torch.long)
        dist.broadcast(lens_split, src=0)
        draft_len = torch.sum(lens_split).item()
        # tree_pos_ids
        tree_pos_ids = torch.zeros(draft_len, dtype=torch.long)
        dist.broadcast(tree_pos_ids, src=0)
        # tree_mask
        mask_shape = (1, 1, draft_len, draft_len)
        tree_mask = torch.zeros(mask_shape, dtype=torch.float32)
        # print(f'stage {stage_model.stage} tree_mask: {tree_mask.shape}, dtype: {tree_mask.dtype}')
        dist.broadcast(tree_mask, src=0)
        tree_mask.to(stage_model.stage_base_model.device)

    tree_pos_ids_split = tree_pos_ids.split(lens_split.tolist(), dim=0)
    cum_lens_split = torch.cumsum(lens_split, dim=-1)
    # [step1] end

    # [step2] start pipelined verification
    isend_task = None
    for i, subseq_len in enumerate(lens_split):
        if i == 0:
            # isend_mask.wait()
            tree_mask_split = tree_mask[..., :cum_lens_split[i], :cum_lens_split[i]].contiguous()
        else:
            tree_mask_split = tree_mask[..., cum_lens_split[i-1]:cum_lens_split[i], :cum_lens_split[i]].contiguous()
        stage_model.stage_base_model.model.tree_mask = tree_mask_split
        if stage_model.is_first_stage:
            seq_split = draft_seqs_split[i]
            # pos_ids_split = tree_pos_ids_split[i] + input_ids.size(-1)
            outputs, sub_hidden_state = stage_model(
                input_ids=seq_split,
                past_key_values=stage_past_key_values,
                position_ids=tree_pos_ids_split[i],
                # tree_mask_range=cum_lens_split[i]  # todo: get partial tree mask in forward of stage model
            )
            hidden_state_comm = sub_hidden_state.cpu()
            hidden_state_splits.append(hidden_state_comm)
            if i == 0:
                isend_mask.wait()
            else:
                if isend_task is not None:
                    isend_task.wait()  # todo: set the timeout in config
            isend_task = dist.isend(hidden_state_comm, dst=config.next_rank)
            # recv_tasks.append(dist.irecv(sub_hidden_state, src=config.last_rank))

        else:
            last_hidden_state = torch.zeros((1, subseq_len, config.hidden_size), dtype=torch.float16)  # todo: d_model?
            dist.recv(last_hidden_state, src=config.last_rank)

            # if not stage_model.is_last_stage:
            # middle stage
            outputs, sub_hidden_state = stage_model(
                inputs_embeds=last_hidden_state.to(device),
                past_key_values=stage_past_key_values,
                position_ids=tree_pos_ids_split[i],
                # tree_mask_range=cum_lens_split[i],
            )
            hidden_state_comm = sub_hidden_state.cpu()
            if isend_task is not None:
                isend_task.wait()
            isend_task = dist.isend(hidden_state_comm, dst=config.next_rank)

    # [step2] end
    if isend_task is not None:
        isend_task.wait()

    # [step3] get complete tree_logits on the first stage
    if stage_model.is_first_stage:
        for i, hidden_state_split in enumerate(hidden_state_splits):
            dist.recv(hidden_state_split, src=config.last_rank)
            # if i == 0:
                # print(f'sub_hidden_state\n{hidden_state_split}')
            hidden_state_splits[i] = hidden_state_split.to(device)
            tree_logits_split = tree_logits_split + (stage_model.stage_base_model.lm_head(hidden_state_splits[i]),)

        tree_logits = torch.concat(tree_logits_split, dim=-2)  # concat at the seq dimension
        # logits = tree_logits[0, retrieve_indices]
        hidden_state = torch.concat(hidden_state_splits, dim=-2)
        return tree_logits, hidden_state


def stage_tree_decoding(
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
        new_token=None,
        hidden_state_new=None,
        sample_p=None,
):
    if model.is_first_stage:
        prev_input_len = torch.tensor(input_ids.shape[1], dtype=torch.int64, device=model.stage_base_model.device)
        # dist.broadcast(prev_input_len, src=0)
        broadcast(prev_input_len, src=0)
        # print(f"best_candidate device: {best_candidate.device}")
        # print(f"accept_length device: {accept_length.device}")
        # print(f"prev_input_len device: {prev_input_len.device}")
        retrieve_indices = retrieve_indices.to(model.stage_base_model.device)
        # print(f"retrieve_indices device: {retrieve_indices.device}")
        
        select_indices = (retrieve_indices[best_candidate, :accept_length+1] + prev_input_len)
        # select_indices_shape = torch.tensor(select_indices.shape, dtype=torch.int64, device=model.stage_base_model.device)
        # dist.broadcast(select_indices_shape, src=0)
        # dist.broadcast(select_indices, src=0)
        broadcast(select_indices, src=0)
    

        # Append the tokens from the best candidate to the input sequence
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
        )
    else:
        prev_input_len = torch.empty((), dtype=torch.int64, device=model.stage_base_model.device)
        # dist.broadcast(prev_input_len, src=0)
        prev_input_len = broadcast(src=0, data_type=torch.int64, shape_length=0)
        
        # select_indices_shape = torch.zeros(1, dtype=torch.int64, device=model.stage_base_model.device)
        # dist.broadcast(select_indices_shape, src=0)
        # select_indices = torch.zeros(select_indices_shape, dtype=torch.int64, device=model.stage_base_model.device)
        # dist.broadcast(select_indices, src=0)
        select_indices = broadcast(src=0, data_type=torch.int64, shape_length=1)

    # Update the past key values based on the selected tokens
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    # print(f"current_length_data shape: {current_length_data.shape}")
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    if model.is_first_stage:
        retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
        prob = sample_p
        if logits_processor is not None:
            token = torch.multinomial(prob, 1)
            token = token[None]
        else:
            token = torch.argmax(prob)
            token = token[None, None]
        # print(f"stage {model.stage} token: {token}")
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
            head=model.stage_base_model.lm_head, logits_processor=logits_processor
        )
        new_token += accept_length + 1
        # print(f"stage {model.stage} draft_tokens: {draft_tokens}")
        # print(f"stage {model.stage} input_ids: {input_ids}")
        # print(f"-------------------------------------\n -------------------------------------")
        return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token


# [ADD] for continuous speculation
def token_tree_partition(draft_tokens, retrieve_indices, total_stage):
    """
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

    tokens_split, lens_split = split_sequence_close_equal_len(draft_tokens, total_stage)
    # tree_position_ids_split = tree_position_ids.split(lens_split)

    cum_seq_lens = torch.cumsum(lens_split, dim=-1)
    # tree_mask_cum = [tree_mask[..., :cum_len, :] for cum_len in cum_seq_lens]

    ri_depth_cum = torch.zeros(retrieve_indices.size(0), dtype=torch.long)
    bottom = torch.full((retrieve_indices.size(0),), -1, dtype=torch.long)
    retrieve_indices = torch.cat((retrieve_indices, bottom[:, None]), dim=1)  # add -1 to bottom to prevent overflow
    subseq_ri_cum_depths = []

    # print(retrieve_indices)
    # print(lens_split)
    for i, cum_seq_len in enumerate(cum_seq_lens):
        for j in range(0 if i == 0 else cum_seq_lens[i - 1], cum_seq_len):
            row_indices = torch.arange(retrieve_indices.size(0), dtype=torch.int)
            cum_ri_leaves = retrieve_indices[row_indices, ri_depth_cum]
            ri_depth_cum[cum_ri_leaves == j] += 1

        # print(ri_depth_cum)
        subseq_ri_cum_depths.append(ri_depth_cum.clone())

    return tokens_split, lens_split, torch.stack(subseq_ri_cum_depths, dim=0)


def fill_pipeline_stages(
        stage_model,
        stage_past_key_values,
        input_ids=None,
        lens_split=None,
        draft_tokens=None,
        retrieve_indices=None,
        tree_mask=None,
        tree_pos_ids=None,
        subseq_ri_cum_depths=None,
):
    # print('============fill_pipeline_stages============')
    config = stage_model.config
    device = stage_model.stage_base_model.device

    # sync lens_split for draft tokens
    if config.is_first_stage:
        isend_tasks = []
        assert lens_split is not None
        isend_tasks.append(dist.broadcast(lens_split, src=0, async_op=True))
        # dist.broadcast(lens_split, src=0)

        isend_tasks.append(dist.broadcast(tree_pos_ids, src=0, async_op=True))
        isend_tasks.append(dist.broadcast(tree_mask, src=0, async_op=True))
        # dist.broadcast(tree_pos_ids, src=0)
        # dist.broadcast(tree_mask, src=0)
        draft_tokens_split = draft_tokens.split(lens_split.tolist(), dim=-1)
        # print(f'stage{config.stage} send_tasks to all stages')

    else:
        lens_split = torch.zeros(config.total_stage, dtype=torch.long)
        dist.broadcast(lens_split, src=0)
        draft_len = torch.sum(lens_split).item()
        tree_pos_ids = torch.zeros(draft_len, dtype=torch.long)
        dist.broadcast(tree_pos_ids, src=0)
        tree_pos_ids = tree_pos_ids.to(device)
        mask_shape = (1, 1, draft_len, draft_len)
        tree_mask = torch.zeros(mask_shape, dtype=torch.float32)
        dist.broadcast(tree_mask, src=0)
        tree_mask = tree_mask.to(device)
        # test_tree_mask = torch.sum(tree_mask[0, 0, ...], dim=-1).to(torch.long)-1
        # test_position_ids = tree_pos_ids - tree_pos_ids[0]
        # print(f'stage{config.stage} recv data from stage0')

    cum_lens_split = torch.cumsum(lens_split, dim=-1)
    tree_pos_ids_split = tree_pos_ids.split(lens_split.tolist(), dim=0)
    
    # isend tree to the last stage
    if config.is_first_stage:
        for isend_task in isend_tasks:
            isend_task.wait()
        # isend_tasks = []
        # print(draft_tokens.device, retrieve_indices.device, subseq_ri_cum_depths.device)
        isend_tasks.append(dist.isend(draft_tokens.cpu(), dst=config.total_stage - 1))  # draft tokens in cuda
        ri_shape = torch.tensor(retrieve_indices.shape, dtype=torch.long)
        isend_tasks.append(dist.isend(ri_shape, dst=config.total_stage - 1))
        isend_tasks.append(dist.isend(retrieve_indices, dst=config.total_stage - 1))  # todo: maybe merge retrieve_indices and subseq_ri_cum_depths and send once
        isend_tasks.append(dist.isend(subseq_ri_cum_depths, dst=config.total_stage - 1))
        # print(f'stage{config.stage} start isend_tasks to the last stage')

    if config.is_last_stage:
        draft_tokens = torch.zeros(1, draft_len, dtype=torch.long)
        dist.recv(draft_tokens, src=0)  # todo: maybe to cuda()
        ri_shape = torch.zeros(2, dtype=torch.long)
        dist.recv(ri_shape, src=0)
        retrieve_indices = torch.zeros(*ri_shape, dtype=torch.long)
        dist.recv(retrieve_indices, src=0)
        subseq_ri_cum_depths = torch.zeros(config.total_stage, ri_shape[0], dtype=torch.long)
        dist.recv(subseq_ri_cum_depths, src=0)
        # print(f'stage{config.stage} recv tree from stage1')
    
    # fill the pipeline stages
    # stage0 doesn't need to recv anything
    isend_task = None
    for i in range(config.total_stage - config.stage):
        # print(config.stage, f'i={i}')
        if i == 0:
            # isend_mask.wait()
            tree_mask_split = tree_mask[..., :cum_lens_split[i], :cum_lens_split[i]]
        else:
            tree_mask_split = tree_mask[..., cum_lens_split[i-1]:cum_lens_split[i], :cum_lens_split[i]]
        
        # set the tree mask for the current stage
        stage_model.stage_base_model.model.tree_mask = tree_mask_split.contiguous()
        subseq_pos_ids = tree_pos_ids_split[i]

        if config.is_first_stage:
            subseq_ids = draft_tokens_split[i]
            
            outputs, sub_hidden_state = stage_model(
                input_ids=subseq_ids,
                past_key_values=stage_past_key_values,
                position_ids=subseq_pos_ids,
            )
        else:
            last_hidden_state = torch.zeros((1, lens_split[i], config.hidden_size), dtype=torch.float16)
            dist.recv(last_hidden_state, src=config.last_rank)
            last_hidden_state = last_hidden_state.to(device)
            outputs, sub_hidden_state = stage_model(
                inputs_embeds=last_hidden_state,
                past_key_values=stage_past_key_values,
                position_ids=subseq_pos_ids,
            )
        if i == 0:
            if config.is_first_stage:
                for t in isend_tasks:
                    t.wait()
                isend_tasks = None
        else:
            if isend_task is not None:
                isend_task.wait()
        if i < config.total_stage - config.stage - 1:  # not send for the last time
            isend_task = dist.isend(sub_hidden_state.cpu(), dst=config.next_rank)
        
    if config.is_first_stage:
        return sub_hidden_state
    if config.is_last_stage:
        return sub_hidden_state, lens_split, draft_tokens, retrieve_indices, tree_mask, tree_pos_ids, subseq_ri_cum_depths
    # middle stages
    return sub_hidden_state, lens_split, tree_mask, tree_pos_ids
    

# generated by Qwen2.5Max
def get_subtree_retrieve_indices(retrieve_indices, cum_depth):
    """
    Get the retrieve_indices of a subtree according to the cumulate depth of each path
    :param retrieve_indices: (path * max_tree_depth)
    :param cum_depth: (path,) < max_tree_depth each
    :return:
    """
    paths, depth = retrieve_indices.shape
    # print(retrieve_indices)
    # print(cum_depth)
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


def pruning(draft_tokens, retrieve_indices, best_candidate, accept_len, new_token, subseq_ri_cum_depths):
    """
    Pruning the token tree on the retrieve_indices
    :param retrieve_indices:
    :param best_candidate:
    :param accept_len:
    :return:
    """
    # todo: what happens when accept_len==0?
    # print(f'best_candidate={best_candidate}, accept_len={accept_len}')
    accepted_indices = retrieve_indices[best_candidate, :accept_len]
    # print(f'accepted_indices={accepted_indices}')

    # judge whether the global leaf node is reached
    if accept_len == retrieve_indices.size(-1) or retrieve_indices[best_candidate, accept_len] == -1:
        # truncate: reach the global leaf
        # print('leaf has been reached')
        return accepted_indices

    # judge whether the new token follows the tree
    matched_candidates = find_prefix_match(retrieve_indices, accepted_indices)
    # print(f'matched_candidates={matched_candidates}')
    next_indices_draft = retrieve_indices[matched_candidates, accept_len]
    # print(f'next_indices_draft={next_indices_draft}')
    next_tokens_draft = draft_tokens[0, next_indices_draft]
    # found the paths with prefix of "accept_tokens + new_token"
    # print(f'next_tokens_draft={next_tokens_draft}')
    # print(f'new_token={new_token}')
    same_indices = torch.nonzero(next_tokens_draft == new_token.cpu()).squeeze(1)
    # print(f'same_indices={same_indices}')
    if same_indices.numel() == 0:
        # truncate: unmatched token
        # print('no match token found in the tree')
        return accepted_indices

    # pruning
    left_candidates = matched_candidates[same_indices]
    left_retrieve_indices = retrieve_indices[left_candidates, accept_len:]  # todo: left_retrieve_indices all -1
    # print(f'retrieve_indices={retrieve_indices}')
    # print(f'left_retrieve_indices={left_retrieve_indices}')
    # print((left_retrieve_indices != -1).sum(dim=1))

    # update: retrieve_indices may be larger than draft_tokens
    # left_indices_global: for the whole tree
    # left_indices: for the tree in pipeline
    max_depth = (left_retrieve_indices != -1).sum(dim=1).max().item()
    left_retrieve_indices = left_retrieve_indices[:, :max_depth]  # drop the all -1 parts
    left_indices_global = process_retrieve_indices(left_retrieve_indices)  # for pruning for other stages
    left_indices_global = torch.cat((accepted_indices, left_indices_global), dim=0)
    left_indices_from_zero = torch.arange(left_indices_global.size(-1) - accept_len, dtype=torch.long)

    left_indices = left_indices_global[left_indices_global < draft_tokens.size(-1)]
    left_draft_tokens = draft_tokens[:, left_indices[accept_len:]]

    # cut the subseq_ri_cum_depths
    left_ri_cum_depths = subseq_ri_cum_depths[1:, left_candidates] - accept_len  # if two same cum_depths, mean the latter is cut completely
    # todo: what happen when only one path is left? Or is it possible?

    # map the left_retrieve_indices to left_indices_from_zero
    transformed_ri = map_retrieve_indices(left_retrieve_indices, left_indices_global[accept_len:], left_indices_from_zero)

    return left_draft_tokens, transformed_ri, left_indices, left_ri_cum_depths


def first_stage_pruning(left_indices, accept_len, draft_tokens, retrieve_indices, subseq_ri_cum_depths=None):
    # print(f'left_indices={left_indices}, accept_len={accept_len}, draft_tokens={draft_tokens.shape}, retrieve_indices={retrieve_indices.shape}, subseq_ri_cum_depths={subseq_ri_cum_depths.shape}')
    # left_draft_indices = left_indices[accept_len:]
    # print(left_indices, draft_tokens)
    prefix_indices = left_indices[:accept_len+1]  # including the new token
    # print(f'stage0 prefix_indices={prefix_indices}, draft_tokens={draft_tokens.shape}')
    prefix_tokens = draft_tokens[:, prefix_indices]  # [accepted tokens + new token]
    accepted_tokens = draft_tokens[:, left_indices[:accept_len]]

    # prune retrieve_indices
    matched_candidates = find_prefix_match(retrieve_indices, prefix_indices)
    left_retrieve_indices = retrieve_indices[matched_candidates, accept_len:]
    first_stage_left_indices = process_retrieve_indices(left_retrieve_indices)


    # # prune draft_tokens
    left_draft_tokens = draft_tokens[:, first_stage_left_indices]

    

    # print(f'stage0 left_retrieve_indices:{left_retrieve_indices}')
    max_depth = (left_retrieve_indices != -1).sum(dim=1).max().item()
    left_retrieve_indices = left_retrieve_indices[:, :max_depth]  # drop the all -1 layers
    left_indices_from_zero = torch.arange(first_stage_left_indices.size(-1), dtype=torch.long)
    transformed_ri = map_retrieve_indices(left_retrieve_indices, first_stage_left_indices, left_indices_from_zero)

    first_stage_left_indices = torch.cat((prefix_indices[:-1], first_stage_left_indices), dim=0)

    # prune subseq_ri_cum_depths
    if subseq_ri_cum_depths is not None:  # continuous
        left_ri_cum_depths = subseq_ri_cum_depths[1:, matched_candidates] - accept_len
        return left_draft_tokens, transformed_ri, accepted_tokens, left_ri_cum_depths, first_stage_left_indices

    # pruned-only
    return left_draft_tokens, transformed_ri, accepted_tokens
    # , prefix_tokens[:, -1]


def token_pruning(
        past_key_values_data_list,
        current_length_data,
        lens_split,
        hidden_state,
        tree_mask,
        tree_pos_ids,
        left_indices,
        global_accept_len,
        accept_len,
        stage
):
    """
    prune the tokens related data: kv-cache, hidden_state, tree_mask
    """
    cur_kv_len = current_length_data[0].item()
    # print(f'stage{stage}, cur_kv_len={cur_kv_len}')
    # print(f'stage{stage}, global_accept_len={global_accept_len}, accept_len={accept_len}')
    cache_device = past_key_values_data_list[0][0].device

    # prune cache
    # ***这里的global_accept_len还是没有加上accept_len长度的
    # print(f'stage{stage}, left_indices={left_indices}')
    left_indices_global = left_indices + global_accept_len
    # print(f'stage{stage}, left_indices_global={left_indices_global}')
    left_indices_in_cache = (left_indices_global[left_indices_global < cur_kv_len]).to(cache_device)
    # print(f'stage{stage}, left_indices_in_cache={left_indices_in_cache}')
    # 对应复制过去，并且修改
    left_indices_in_cache_size = left_indices_in_cache.size(-1)
    # left_draft_size = left_indices_global.size(-1)
    for past_key_values_data in past_key_values_data_list:
        left_kv_cache = past_key_values_data[..., left_indices_in_cache, :]
        cache_dst = past_key_values_data[..., global_accept_len:global_accept_len+left_indices_in_cache_size, :]
        cache_dst.copy_(left_kv_cache, non_blocking=True)
    current_length_data.fill_(global_accept_len + left_indices_in_cache_size)

    # prune lens_split
    if lens_split is not None:
        # test kv_len
        assert cur_kv_len == global_accept_len + torch.sum(lens_split[:4-stage]), f'stage{stage} wrong kv_len={cur_kv_len} while global_accept={global_accept_len} and lens_split={lens_split}'

        cum_lens = torch.cumsum(lens_split, dim=0)
        lens_split = torch.tensor([torch.sum((left_indices >= cum_lens[i-1]) & (left_indices < cum_lens[i])) for i in range(1, cum_lens.size(-1))], dtype=torch.long)

    # prune hidden_state
    # the hidden_state here are the output of the current stage_model
    if hidden_state is not None:
        cur_hs_len = hidden_state.size(-2)
        # print(f'stage{stage}, hiddent_state={hidden_state.size()}')
        hs_indices_start_idx = cur_kv_len - cur_hs_len  # [update]: idx should minus 1 additionally
        left_indices_in_hs = left_indices_in_cache[(left_indices_in_cache >= hs_indices_start_idx) & (left_indices_in_cache < cur_kv_len)] - hs_indices_start_idx
        left_indices_in_hs = left_indices_in_hs.to(hidden_state.device)
        # print(f'stage{stage}, hs_indices_start_idx={hs_indices_start_idx}, cur_kv_len={cur_kv_len}, left_indices_in_hs={left_indices_in_hs}')
        # print(f'stage{stage}, hidden_state: {hidden_state.device}, shape: {hidden_state.shape}')  # todo: why left_indices_in_hs can be on cuda?
        # print(f'stage{stage}, left_indices_in_hs: {left_indices_in_hs.device}, shape: {left_indices_in_hs.shape}')
        if left_indices_in_hs.numel() > 0:
            assert torch.max(left_indices_in_hs) < hidden_state.size(-2), f'stage{stage} left_indices_in_hs={left_indices_in_hs} is out of range'
        hidden_state = hidden_state[..., left_indices_in_hs, :]  # todo: bug occurs when hidden_state is empty tensor

    # prune tree_mask
    if tree_mask is not None:
        tree_mask_left_indices = left_indices[accept_len:]
        # print(f'stage{stage}, tree_mask_left_indices={tree_mask_left_indices}')
        # print('before', tree_mask.shape)
        assert torch.max(tree_mask_left_indices) < tree_mask.size(-1), f'stage{stage} tree_mask_left_indices={tree_mask_left_indices} is out of range'
        tree_mask = tree_mask[..., tree_mask_left_indices[:, None], tree_mask_left_indices].contiguous()
        # print('pruned', tree_mask.shape)

    # prune tree_pos_ids
    if tree_pos_ids is not None:
        tree_pos_ids = tree_pos_ids[tree_mask_left_indices]

    return past_key_values_data_list, current_length_data, lens_split, hidden_state, tree_mask, tree_pos_ids


def get_parent_indices(tree_mask: torch.Tensor) -> torch.Tensor:
    """
    Use parent_idx = parent_indices[token_idx] to get the parent of token_idx
    Return the parent indices of each token in the tree
    INPUT: tree_mask, [tree_size, tree_size]
    OUTPUT: parent_indices, [tree_size]
    """
    if tree_mask.dtype != torch.bool:
        tree_mask = tree_mask.to(torch.bool)

    # get lower triangular mask to avoid including the token itself
    n = tree_mask.size(0)
    offset = torch.tril(torch.ones(n, n, dtype=torch.bool, device=tree_mask.device), diagonal=-1)

    masked = tree_mask & offset  # keep the valid parent nodes
    
    # reverse the masked matrix to find the last true value in each row
    reversed_masked = torch.flip(masked, dims=[1])
    max_values, parent_indices = torch.max(reversed_masked, dim=1)
    
    # convert the indices and handle the invalid values
    parent_indices = (n - 1 - parent_indices)  # reverse the indices
    parent_indices[~max_values] = -1  # if all are false, set to -1
    return parent_indices


def merge_two_tree(
        tree1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tree2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        lens_split,
        subseq_ri_cum_depths
):
    """
    Merge two tree that share the same root node, tree1 is the old tree, tree2 is the new tree
    The merged tree has the draft_tokens of: {draft_tokens1, added_tokens}
    """
    # print(f'===========start merging two tree')

    draft_tokens1, retrieve_indices1, tree_mask1, tree_pos_ids1 = tree1
    draft_tokens2, retrieve_indices2, tree_mask2, tree_pos_ids2 = tree2
    # print(f'tree_size: {draft_tokens1.size(-1)}, tree2_size: {draft_tokens2.size(-1)}')
    # print(f'lens_split: {lens_split}')

    # token_tree1 = map_retrieve_indices(retrieve_indices1, torch.arange(draft_tokens1.size(-1)), draft_tokens1[0, :])
    # token_tree2 = map_retrieve_indices(retrieve_indices2, torch.arange(draft_tokens2.size(-1)), draft_tokens2[0, :])
    # print(f'token_tree1: {token_tree1}')
    # print(f'token_tree2: {token_tree2}')

    tree1_depth = (retrieve_indices1 != -1).sum(dim=1).max()  # maybe contain -1
    # print(f'tree1_depth: {tree1_depth}')
    tree2_depth = retrieve_indices2.size(1)
    tree_mask1 = tree_mask1[0, 0, ...]
    tree_mask2 = tree_mask2[0, 0, ...]
    tree1_size = draft_tokens1.size(-1)
    # print(f'tree1_size: {tree1_size}')
    # print(f'tree_mask1: {tree_mask1.shape}')

    # get paths of draft_tokens1
    paths_tree1_idx = {}
    for i, draft_token in enumerate(draft_tokens1[0, :]):
        # important: squeeze(1) to avoid the shape (1) and use tuple(int) for the dict key
        token_path = draft_tokens1[0, torch.nonzero(tree_mask1[i, :]).squeeze(1)].tolist()
        paths_tree1_idx[tuple(token_path)] = i


    # [merge draft_tokens]
    # init draft_tokens_merged as draft_tokens1
    draft_tokens_merged = draft_tokens1.squeeze(0).clone()  # 转成1维处理先
    # [merge tree_pos_ids]
    # init merged_tree_pos_ids as tree_pos_ids1
    merged_tree_pos_ids = tree_pos_ids1.clone()
    # print(f'merged_tree_pos_ids: {merged_tree_pos_ids}')

    # go through draft_tokens2
    paths_tree2 = set()
    index_mapping_2_to_merged = torch.zeros(draft_tokens2.size(1), dtype=torch.long)
    for i, draft_token in enumerate(draft_tokens2[0, :]):
        token_path = tuple(draft_tokens2[0, torch.nonzero(tree_mask2[i, :]).squeeze(1)].tolist())
        # print('token_path: ', token_path)
        paths_tree2.add(token_path)

        if len(token_path) <= tree1_depth and token_path in paths_tree1_idx:
            # print('Same token from tree1 and tree2: ', token_path)
            index_mapping_2_to_merged[i] = paths_tree1_idx[token_path]
        else:
            mapped_idx = draft_tokens_merged.size(0)

            # expand draft_tokens
            draft_tokens_merged = torch.cat((draft_tokens_merged, draft_tokens2[:, i]), dim=0)
            # expand tree_pos_ids
            merged_tree_pos_ids = torch.cat((merged_tree_pos_ids, tree_pos_ids2[i].unsqueeze(0)), dim=0)

            # update index_mapping
            index_mapping_2_to_merged[i] = mapped_idx
    # todo: 优化：遍历时到了比tree1更深的点应该可以批量化merge
    # [merge draft_tokens] finish

    # print(f'draft_tokens_merged: {draft_tokens_merged.shape}')
    # print(f'merged_tree_pos_ids: {merged_tree_pos_ids.shape}')
    
    # print(index_mapping_2_to_merged)

    # [merge tree_mask]
    merged_size = draft_tokens_merged.size(0)
    # print(f'merged_size: {merged_size}')
    # init merged_tree_mask as tree_mask1
    merged_tree_mask = torch.zeros(
        (merged_size, merged_size),  # notice: 2-d here
        dtype=tree_mask1.dtype,
        device=tree_mask1.device
    )
    merged_tree_mask[:tree1_size, :tree1_size] = tree_mask1
    parent_indices = get_parent_indices(tree_mask2)
    # print(f'parent_indices: {parent_indices}')
    for i, draft_token in enumerate(draft_tokens2[0, :]):
        token_path = tuple(draft_tokens2[0, torch.nonzero(tree_mask2[i, :]).squeeze(1)].tolist())

        if len(token_path) <= tree1_depth and token_path in paths_tree1_idx:
            index_mapping_2_to_merged[i] = paths_tree1_idx[token_path]
        else:
            mapped_idx = index_mapping_2_to_merged[i]

            # expand tree_mask
            parent_idx_mapped = index_mapping_2_to_merged[parent_indices[i]]
            mapped_parent_mask_row = merged_tree_mask[parent_idx_mapped, :parent_idx_mapped+1]
            merged_tree_mask[mapped_idx, :parent_idx_mapped+1] = mapped_parent_mask_row
            merged_tree_mask[mapped_idx, mapped_idx] = 1

    # [merge tree_mask] finish
    # print(f'merged_tree_mask: {merged_tree_mask}')
    # test_pos_ids = torch.sum(merged_tree_mask, dim=1).to(torch.long) - 1
    # print(f'test_pos_ids: {test_pos_ids}')
    # print(f'merged_tree_pos_ids: {merged_tree_pos_ids}')
    # print(f'Same position ids: {torch.equal(test_pos_ids, merged_tree_pos_ids)}')
    
    # [merge retrieve_indices]
    # get leaf nodes of tree1
    leave_paths1 = {}
    for i in range(retrieve_indices1.size(0)):
        leaf_node_pos = torch.nonzero(retrieve_indices1[i, :] != -1).squeeze(1)[-1]
        # leaf_node_idx = retrieve_indices1[i, leaf_node_pos]
        leaf_path = tuple(draft_tokens1[0, retrieve_indices1[i, :leaf_node_pos+1]].tolist())
        leave_paths1[leaf_path] = i

    # get leaf nodes of tree2
    leave_paths2 = {}
    for i in range(retrieve_indices2.size(0)):
        leaf_node_pos = torch.nonzero(retrieve_indices2[i, :] != -1).squeeze(1)[-1]
        # leaf_node_idx = retrieve_indices2[i, leaf_node_pos]
        leaf_path = tuple(draft_tokens2[0, retrieve_indices2[i, :leaf_node_pos+1]].tolist())
        leave_paths2[leaf_path] = i

    selected_leaves1 = torch.zeros(retrieve_indices1.size(0), dtype=torch.bool)
    selected_leaves2 = torch.zeros(retrieve_indices2.size(0), dtype=torch.bool)
    # todo: bug for display with wrong leaf_path
    for leaf_path, leaf_path_idx in leave_paths1.items():
        if leaf_path in paths_tree2 and leaf_path not in leave_paths2:
            # print(leaf_path, 'remove')
            pass
        else:
            # print(leaf_path, 'select')
            selected_leaves1[leaf_path_idx] = True
    for leaf_path, leaf_path_idx in leave_paths2.items():
        if leaf_path not in paths_tree1_idx:
            # print(leaf_path, 'select')
            selected_leaves2[leaf_path_idx] = True
        else:
            # print(leaf_path, 'remove')
            pass

    ri_selected1 = F.pad(retrieve_indices1[selected_leaves1, :], (0, tree2_depth - tree1_depth), value=-1)
    ri_selected2 = retrieve_indices2[selected_leaves2, :]
    ri_selected2 = map_retrieve_indices(ri_selected2, torch.arange(index_mapping_2_to_merged.size(0)), index_mapping_2_to_merged)
    retrieve_indices_merged = torch.cat((ri_selected1, ri_selected2), dim=0)

    # last_subseq_ri_cum_depth = torch.nonzero(retrieve_indices_merged != -1).squeeze(1)[-1]
    # ri_flat = retrieve_indices_merged.view(-1)
    # ri_unique = torch.sort(torch.unique(ri_flat[ri_flat != -1]))
    # print(ri_unique)

    # merged_token_tree = map_retrieve_indices(retrieve_indices_merged, torch.arange(draft_tokens_merged.size(0)), draft_tokens_merged)
    # print(f'merged_token_tree: {merged_token_tree}')
    # [merge retrieve_indices] finish

    # todo: update subseq_ri_cum_depths and lens_split
    lens_split = torch.cat((lens_split, torch.tensor([draft_tokens_merged.size(0) - tree1_size], dtype=torch.long)))
    # print(f'lens_split: {lens_split}')

    n_leaves = retrieve_indices_merged.size(0)
    subseq_ri_cum_depths = []
    cum_seq_lens = torch.cumsum(lens_split[:-1], dim=0)
    bottom = torch.full((n_leaves,), -1, dtype=torch.long)
    retrieve_indices_filled = torch.cat((retrieve_indices_merged, bottom[:, None]), dim=1)  # add -1 to bottom to prevent overflow
    # print(retrieve_indices)
    # print(lens_split)
    ri_depth_cum = torch.zeros(n_leaves, dtype=torch.long)
    for i, cum_seq_len in enumerate(cum_seq_lens):
        for j in range(0 if i == 0 else cum_seq_lens[i - 1], cum_seq_len):
            row_indices = torch.arange(n_leaves, dtype=torch.int)
            cum_ri_leaves = retrieve_indices_filled[row_indices, ri_depth_cum]
            ri_depth_cum[cum_ri_leaves == j] += 1

        # print(ri_depth_cum)
        # update: 只计算到在pipeline里的draft token tree部分，即将输入的最新一段单独算
        subseq_ri_cum_depths.append(ri_depth_cum.clone())
        # todo: 优化最后一段直接append，不需要累加

    # print(f'subseq_ri_cum_depths: {subseq_ri_cum_depths}')
    
    return draft_tokens_merged.unsqueeze(0), retrieve_indices_merged, merged_tree_mask[None, None], merged_tree_pos_ids, lens_split, torch.stack(subseq_ri_cum_depths, dim=0)




# def grow_token_tree(
#         stage_model,
#         input_ids_for_draft,
#         left_hidden_state,
#         left_draft_tokens,
#         left_retrieve_indices,
#         left_tree_mask,
#         left_tree_pos_ids,
#         accept_len,
#         logits_processor
# ):
#     draft_tokens, retrieve_indices, tree_mask, tree_position_ids = stage_model.ea_layer.topK_genrate(
#         left_hidden_state,
#         input_ids=input_ids_for_draft,
#         head=stage_model.base_model.lm_head,
#         logits_processor=logits_processor
#     )


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
        accept_length = candidates_accept_length.max()
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
            gt_logits = logits_processor(None, gt_logits)
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
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        # print(f'evaluate_posterior: accept_cand={accept_cand}')
        return torch.tensor(best_candidate), accept_length - 1, sample_p


if __name__ == '__main__':
    sequence = torch.randint(0, 10000, (1, 209), dtype=torch.long)
    res = split_sequence_close_equal_len(sequence, 6)
    for i in res:
        print(i)
