# reference: https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/utils.py - def evaluate_posterior()
# codes with comment "EAGLE" are original EAGLE codes

import random
import copy
import time
import torch
from typing import Union, Iterable, List
import torch.distributed as dist
# from stage_ea_model import StageEaModel
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from tools.communicator import*

TOPK = 10  # topk for sparse tree

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
        return [base_size if i >= reminder else base_size+1 for i in range(n)]


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
    split_lens = torch.tensor(split_lens, dtype=torch.int32)
    return split_seqs, split_lens


# [ADD] for pipeline
# def pipeline_prefill(stage_model, input_ids, stage_past_key_values=None, logits_processor=None, tree_mask=None):
#     config = stage_model.config
#     total_stage = config.total_stage
#     if stage_model.is_first_stage:
#         hidden_state_split = ()
#         recv_tasks = []
#     isend_task = None
#     if input_ids.size(-1) > 20:  # pipelined prefill
#         seq_splits, lens_split = split_sequence_close_equal_len(input_ids, total_stage)
#         for subseq_len in lens_split:
#             if stage_model.is_first_stage:
#                 sub_hidden_state = stage_model(
#                     input_ids=subseq_len,
#                     past_key_values=stage_past_key_values,
#                 )
#                 hidden_state_split = hidden_state_split + (sub_hidden_state,)
#                 if isend_task is not None:
#                     isend_task.wait()
#                 isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)
#                 irecv_task = dist.irecv(sub_hidden_state)
#                 recv_tasks.append(irecv_task)
#             else:
#                 last_hidden_state = torch.zeros((1, subseq_len, config.d_model), dtype=torch.float16)
#                 dist.recv(last_hidden_state, src=config.last_rank)

#                 # if not stage_model.is_last_stage:
#                 # middle stage
#                 sub_hidden_state = stage_model(
#                     last_hidden_state=last_hidden_state,
#                     past_key_values=stage_past_key_values,
#                 )
#                 if isend_task is not None:
#                     isend_task.wait()
#                 isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)

#         if stage_model.is_first_stage:
#             for i, recv_task in enumerate(recv_tasks):
#                 recv_task.wait()
#                 hidden_state_split = hidden_state_split + (hidden_state_split[i],)

#             hidden_state = torch.concat(hidden_state_split, dim=-2)
#             return hidden_state

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
        orig, hidden_states = prefill_pipeline(stage_model, past_key_values, input_ids)
        
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
        prefill_pipeline(stage_model = stage_model, stage_past_key_values = past_key_values)

# [ADD] for pipeline
def tree_partition_pipeline(draft_tokens, tree_position_ids, tree_mask, total_stage: int):
    """
    split the input tree into multiple subtrees
    :param total_stage:
    :param draft_tokens: flattened tree token sequence following the causal rule
    :param tree_position_ids:
    :return: Tuple(subtree1=(draft_tokens, tree_position_ids, tree_attention_mask, retrieve_indices),...)
    """
    seqs_split, lens_split = split_sequence_close_equal_len(draft_tokens, total_stage)
    tree_pos_ids_split = tree_position_ids.split(tuple(lens_split))
    # print(f"lens_split={lens_split}")
    cum_seq_lens = torch.cumsum(lens_split, dim=-1)
    
    # split tree mask for seqs_split respectively
    tree_mask_split = []
    for i in range(len(lens_split)):
        if i > 0: # make sure the tree mask is contiguous, could be optimized
            tree_mask_split.append(tree_mask[..., cum_seq_lens[i-1]:cum_seq_lens[i], :cum_seq_lens[i]].contiguous())
        else:
            tree_mask_split.append(tree_mask[..., :cum_seq_lens[i], :cum_seq_lens[i]].contiguous())
    
    return seqs_split, tree_pos_ids_split, tree_mask_split, lens_split


# [MODIFIED] from tree_decoding()
# def stage_tree_decoding(
#         stage_model,
#         stage_past_key_values=None,
#         retrieve_indices=None,
#         draft_seqs_split=None,
#         tree_pos_ids_split=None,
#         lens_split=None,
#         input_ids=None,
# ):
#     """
#     pipelined tree decoding for verification
#     :param stage_model: necessary
#     :param stage_past_key_values: necessary
#     :param draft_seqs_split: only necessary for stage1
#     :param tree_pos_ids_split: only necessary for stage1
#     :param lens_split: only necessary for stage1
#     :param input_ids: only necessary for stage1
#     :param retrieve_indices: only necessary for stage(-1)
#     :return: return result on stage1
#     """
#     # [step1] prepare necessary data to all devices
#     # todo: overlap with the computation of the 1st subseq on the 1st stage
#     config = stage_model.config
#     if stage_model.is_first_stage:
#         assert None not in (draft_seqs_split, tree_pos_ids_split, input_ids, retrieve_indices)
#         dist.broadcast(lens_split, src=0)
#         # ri_shape = torch.tensor(retrieve_indices.shape, dtype=torch.int32)
#         # dist.send(ri_shape, dst=config.last_rank)
#         # dist.send(retrieve_indices, dst=config.last_rank)
#         recv_tasks = []
#         tree_logits_split = ()
#         hidden_state_split = ()

#     else:
#         lens_split = torch.zeros(stage_model.total_stage, dtype=torch.int32)
#         dist.broadcast(lens_split, src=0)

#     cum_lens_split = torch.cumsum(lens_split, dim=-1)
#     # [step1] end

#     # [step2] start pipelined verification
#     isend_task = None
#     irecv_task = None
#     for i, subseq_len in enumerate(lens_split):
#         if stage_model.is_first_stage:
#             seq_split = draft_seqs_split[i]
#             pos_ids_split = tree_pos_ids_split[i]
#             sub_hidden_state = stage_model(
#                 input_ids=seq_split,
#                 past_key_values=stage_past_key_values,
#                 position_ids=pos_ids_split,
#                 tree_mask_range=cum_lens_split[i]  # todo: get partial tree mask in forward of stage model
#             )
#             hidden_state_split = hidden_state_split + (sub_hidden_state,)
#             if isend_task is not None:
#                 isend_task.wait()  # todo: set the timeout in config
#             isend_task = dist.isend(sub_hidden_state, dst=config.next_stage)
#             recv_tasks.append(dist.irecv(sub_hidden_state, src=config.last_rank))

#         else:
#             last_hidden_state = torch.zeros((1, subseq_len, config.d_model), dtype=torch.float16)  # todo: d_model?
#             dist.recv(last_hidden_state, src=config.last_rank)

#             # if not stage_model.is_last_stage:
#             # middle stage
#             sub_hidden_state = stage_model(
#                 input_embeds=last_hidden_state,
#                 past_key_values=stage_past_key_values,
#                 tree_mask_range=cum_lens_split[i],
#             )
#             if isend_task is not None:
#                 isend_task.wait()
#             isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)
#     # [step2] end

#     # [step3] get complete tree_logits on the first stage
#     if stage_model.is_first_stage:
#         for i, recv_task in enumerate(recv_tasks):
#             recv_task.wait()
#             tree_logits_split = tree_logits_split + (stage_model.lm_head(hidden_state_split[i]),)

#         tree_logits = torch.concat(tree_logits_split, dim=-2)  # concat at the seq dimension
#         logits = tree_logits[0, retrieve_indices]
#         hidden_state = torch.concat(hidden_state_split, dim=-2)
#         return logits, hidden_state
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
        accept_cand = candidates[0][:1]
        best_candidate = 0
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
        return torch.tensor(best_candidate), accept_length - 1, sample_p


if __name__ == '__main__':
    sequence = torch.randint(0, 10000, (1, 209), dtype=torch.int32)
    res = split_sequence_close_equal_len(sequence, 6)
    for i in res:
        print(i)
