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
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


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


def split_close_equal(total_size, n):
    assert total_size > n > 0
    base_size = total_size // n
    reminder = total_size % n
    if reminder == 0:
        return [base_size for _ in range(n)]
    else:
        # leave the smaller one on the front
        return [base_size if i >= reminder else base_size + 1 for i in range(n)]


def split_sequence_close_equal_len(sequence: torch.Tensor, split_cnt: Union[int, Iterable[int], torch.Tensor]):
    if len(sequence.shape) <= 2:
        seq_len = sequence.size(-1)
    else:
        raise ValueError('Sequence for splitting can not have a batch_size larger than 2.')

    if isinstance(split_cnt, int):
        split_lens = split_close_equal(seq_len, split_cnt)
    else:
        split_lens = split_cnt
    split_lens = torch.tensor(split_lens, dtype=torch.long)

    assert sum(split_lens) == seq_len
    split_seqs = sequence.split(split_lens, dim=-1)
    return split_seqs, split_lens


# [ADD] for pipeline
# collaborative function
def pipeline_prefill(
        stage_model,
        input_ids=None,
        stage_past_key_values=None,
        logits_processor=None,
        tree_mask=None
):
    config = stage_model.config
    stage_base_model = stage_model.stage_base_model
    total_stage = config.total_stage
    if config.is_first_stage:
        hidden_state_splits = ()  # save the hidden_state for later transmission
        # recv_tasks = []
    isend_task = None
    subseq_kv_cache = ()
    if input_ids.size(-1) > 20:  # pipelined prefill, split the sequence when long enough
        seq_splits, lens_split = split_sequence_close_equal_len(input_ids, total_stage)
        for subseq_len in lens_split:
            if config.is_first_stage:
                sub_hidden_state, stage_past_key_values = stage_base_model(
                    input_ids=subseq_len,
                    past_key_values=stage_past_key_values,
                )
                hidden_state_splits = hidden_state_splits + (sub_hidden_state,)
                if isend_task is not None:
                    isend_task.wait()
                isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)

            else:
                last_hidden_state = torch.zeros((1, subseq_len, config.d_model), dtype=torch.float16)
                dist.recv(last_hidden_state, src=config.last_rank)

                # middle stage
                sub_hidden_state, stage_past_key_values = stage_model(
                    last_hidden_state=last_hidden_state,
                    past_key_values=stage_past_key_values,
                )
                if isend_task is not None:
                    isend_task.wait()
                isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)

            subseq_kv_cache = subseq_kv_cache + (stage_past_key_values,)

        stage_past_key_values = torch.cat(subseq_kv_cache, dim=-2)

        if stage_model.is_first_stage:
            origs = ()

            for i, hidden_state_split in enumerate(hidden_state_splits):
                recv_task = dist.irecv(hidden_state_split, src=config.last_rank)
                if i > 0:
                    origs = origs + (stage_base_model.lm_head(hidden_state_splits[i - 1]),)
                recv_task.wait()

            hidden_state = torch.concat(hidden_state_splits, dim=-2)
            orig = torch.concat(origs, dim=-2)

            return orig, hidden_state, stage_past_key_values

        else:
            return stage_past_key_values


# [MODIFIED] from initialize_tree()
# collaborative function
def initialize_tree_pipeline(stage_model, input_ids=None, past_key_values=None, logits_processor=None):
    config = stage_model.config
    if config.is_first_stage:
        assert input_ids is not None
    output = pipeline_prefill(
        stage_model, input_ids=input_ids, stage_past_key_values=past_key_values, output_orig=True
    )

    if config.is_first_stage:
        orig, hidden_states = output

        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(orig[:, -1])
            token = token[None, None]
        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        # Clone the output hidden states

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = stage_model.ea_layer.topK_genrate(
            hidden_states,
            input_ids,
            stage_model.base_model.lm_head,
            logits_processor
        )
        # only return the draft token tree for on the first stage
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token


# [ADD] for pipeline
def tree_partition_pipeline(draft_tokens, tree_position_ids, total_stage: int):
    """
    split the input sequence into multiple subsequences
    :param total_stage:
    :param draft_tokens: flattened tree token sequence following the causal rule
    :param tree_position_ids:
    :return: Tuple(subtree1=(draft_tokens, tree_position_ids, tree_attention_mask, retrieve_indices),...)
    """
    seqs_split, lens_split = split_sequence_close_equal_len(draft_tokens, total_stage)
    tree_pos_ids_split = tree_position_ids.split(lens_split)
    # cum_seq_lens = torch.cumsum(lens_split, dim=-1)

    return seqs_split, tree_pos_ids_split, lens_split


# [MODIFIED] from tree_decoding()
def stage_tree_decoding(
        stage_model,
        stage_past_key_values=None,
        retrieve_indices=None,
        draft_seqs_split=None,
        tree_pos_ids_split=None,
        lens_split=None,
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
    if stage_model.is_first_stage:
        assert None not in (draft_seqs_split, tree_pos_ids_split, input_ids, retrieve_indices)
        dist.broadcast(lens_split)
        # ri_shape = torch.tensor(retrieve_indices.shape, dtype=torch.long)
        # dist.send(ri_shape, dst=config.last_rank)
        # dist.send(retrieve_indices, dst=config.last_rank)
        recv_tasks = []
        tree_logits_split = ()
        hidden_state_split = ()

    else:
        lens_split = torch.zeros(stage_model.total_stage, dtype=torch.long)
        dist.broadcast(lens_split, src=0)

    cum_lens_split = torch.cumsum(lens_split, dim=-1)
    # [step1] end

    # [step2] start pipelined verification
    isend_task = None
    irecv_task = None
    for i, subseq_len in enumerate(lens_split):
        if stage_model.is_first_stage:
            seq_split = draft_seqs_split[i]
            pos_ids_split = tree_pos_ids_split[i]
            sub_hidden_state = stage_model(
                input_ids=seq_split,
                past_key_values=stage_past_key_values,
                position_ids=pos_ids_split,
                tree_mask_range=cum_lens_split[i]  # todo: get partial tree mask in forward of stage model
            )
            hidden_state_split = hidden_state_split + (sub_hidden_state,)
            if isend_task is not None:
                isend_task.wait()  # todo: set the timeout in config
            isend_task = dist.isend(sub_hidden_state, dst=config.next_stage)
            recv_tasks.append(dist.irecv(sub_hidden_state, src=config.last_rank))

        else:
            last_hidden_state = torch.zeros((1, subseq_len, config.d_model), dtype=torch.float16)  # todo: d_model?
            dist.recv(last_hidden_state, src=config.last_rank)

            # if not stage_model.is_last_stage:
            # middle stage
            sub_hidden_state = stage_model(
                last_hidden_state=last_hidden_state,
                past_key_values=stage_past_key_values,
                tree_mask_range=cum_lens_split[i],
            )
            if isend_task is not None:
                isend_task.wait()
            isend_task = dist.isend(sub_hidden_state, dst=config.next_rank)
    # [step2] end

    # [step3] get complete tree_logits on the first stage
    if stage_model.is_first_stage:
        for i, recv_task in enumerate(recv_tasks):
            recv_task.wait()
            tree_logits_split = tree_logits_split + (stage_model.lm_head(hidden_state_split[i]),)

        tree_logits = torch.concat(tree_logits_split, dim=-2)  # concat at the seq dimension
        logits = tree_logits[0, retrieve_indices]
        hidden_state = torch.concat(hidden_state_split, dim=-2)
        return logits, hidden_state


# [MODIFIED] from update_inference_inputs()
@torch.no_grad()
def update_stage_inference_inputs(
        model,
        past_key_values_data_list,
        current_length_data,
        logits_processor,
        input_ids=None,
        candidates=None,
        best_candidate=None,
        accept_length=None,
        retrieve_indices=None,
        new_token=None,
        hidden_state_new=None,
        sample_p=None,
):
    if model.config.is_first_stage:
        prev_input_len = torch.tensor(input_ids.shape[1], dtype=torch.int64)
        dist.broadcast(prev_input_len, src=0)

        select_indices = (retrieve_indices[best_candidate, :accept_length + 1] + prev_input_len)
        select_indices_shape = torch.tensor(select_indices.shape, dtype=torch.int64)
        dist.broadcast(select_indices_shape, src=0)
        dist.broadcast(select_indices, src=0)

        # Append the tokens from the best candidate to the input sequence
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
        )
    else:
        prev_input_len = torch.zeros(1, dtype=torch.int64)
        dist.broadcast(prev_input_len, src=0)

        select_indices_shape = torch.zeros(1, dtype=torch.int64)
        dist.broadcast(select_indices_shape)
        select_indices = torch.zeros(select_indices_shape, dtype=torch.int64)
        dist.broadcast(select_indices, src=0)

    # Update the past key values based on the selected tokens
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
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

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
            head=model.base_model.lm_head, logits_processor=logits_processor
        )
        new_token += accept_length + 1

        return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token


# [ADD] for continuous speculation
def token_tree_partition(draft_tokens, retrieve_indices, tree_position_ids, total_stage):
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
    tree_position_ids_split = tree_position_ids.split(lens_split)

    cum_seq_lens = torch.cumsum(lens_split, dim=-1)
    # tree_mask_cum = [tree_mask[..., :cum_len, :] for cum_len in cum_seq_lens]

    ri_depth_cum = torch.zeros(retrieve_indices.size(-1), dtype=torch.long)
    subseq_ri_cum_depths = []

    for i, cum_seq_len in enumerate(cum_seq_lens):
        for j in range(0 if i == 0 else cum_seq_lens[i - 1], cum_seq_len):
            cum_ri_leaves = retrieve_indices[:, ri_depth_cum]
            ri_depth_cum[cum_ri_leaves == j] += 1

        subseq_ri_cum_depths.append(ri_depth_cum.clone())

    return tokens_split, lens_split, tree_position_ids_split, torch.stack(subseq_ri_cum_depths, dim=0)


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

    match_paths = torch.nonzero(matches).squeeze()

    return match_paths


def process_retrieve_indices(retrieve_indices):
    flattened = retrieve_indices.view(-1)
    mask = flattened != -1
    filtered = flattened[mask]

    unique_values = torch.unique(filtered)
    sorted_values = torch.sort(unique_values).values

    return sorted_values


def map_retrieve_indices(retrieve_indices, a, b):
    # consider a is sorted, transform elements in retrieve_indices by mapping a->b
    flat = retrieve_indices.view(-1)
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
    accepted_indices = retrieve_indices[best_candidate, :accept_len]
    # judge whether the global leaf node is reached
    if accept_len == retrieve_indices.size(-1) or retrieve_indices[best_candidate, accept_len + 1] == -1:
        # reach the global leaf
        return None

    # judge whether the new token follows the tree
    matched_candidates = find_prefix_match(retrieve_indices, accepted_indices)
    next_indices_draft = retrieve_indices[matched_candidates, accept_len+1]
    next_tokens_draft = draft_tokens[0, next_indices_draft]
    # found the paths with prefix of "accept_tokens + new_token"
    same_indices = torch.nonzero(next_tokens_draft == new_token).squeeze()
    if same_indices.numel() == 0:
        # no match token found in the tree
        return None

    # pruning
    left_candidates = matched_candidates[same_indices]
    left_retrieve_indices = retrieve_indices[left_candidates, accept_len:]  # todo: left_retrieve_indices all -1
    max_depth = (left_retrieve_indices != -1).sum(dim=1).max().item()
    left_retrieve_indices = left_retrieve_indices[:, :max_depth]  # drop the all -1 parts
    left_indices = process_retrieve_indices(left_retrieve_indices)  # for pruning for other stages
    left_indices_from_zero = torch.arange(left_indices.size(-1), dtype=torch.long)

    # cut the subseq_ri_cum_depths
    left_ri_cum_depths = subseq_ri_cum_depths[1:, left_candidates] - accept_len  # if two same cum_depths, mean the latter is cut completely
    # todo: what happen when only one path is left? Or is it possible?

    # map the left_retrieve_indices to left_indices_from_zero
    transformed_ri = map_retrieve_indices(left_retrieve_indices, left_indices, left_indices_from_zero)

    return transformed_ri, left_indices, left_ri_cum_depths


def token_pruning(past_key_values, tree_mask, hidden_state, left_indices, global_accept_len, accept_len):
    draft_kv_cache = past_key_values[..., global_accept_len:, :]
    cur_kv_len = past_key_values[0][0].size(-2)
    cur_hs_len = hidden_state.size(-2)

    left_indices_in_cache = left_indices[left_indices < cur_kv_len]
    left_kv_cache = draft_kv_cache[..., left_indices_in_cache, :]
    # 对应复制过去，并且修改
    global_accept_len += accept_len




# verification: use evaluate_posterior() from EAGLE
# collaborative function
def update_token_tree():
    """
    update the token tree across all pipeline stages
    :return:
    """


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
    sequence = torch.randint(0, 10000, (1, 209), dtype=torch.long)
    res = split_sequence_close_equal_len(sequence, 6)
    for i in res:
        print(i)
