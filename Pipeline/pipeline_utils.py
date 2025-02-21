import torch
from typing import Union, Iterable
import torch.distributed as dist
from stage_ea_model import StageEaModel

def split_sequence_close_equal_len(sequence: torch.Tensor, split_cnt: Union[int, Iterable[int], torch.Tensor]):

    if len(sequence.shape) <= 2:
        seq_len = sequence.size(-1)
    else:
        raise ValueError('Sequence for splitting can not have a batch_size larger than 2.')

    if isinstance(split_cnt, int):
        base_length = seq_len // split_cnt
        reminder = seq_len % split_cnt
        split_lens = [base_length + 1 if i < reminder else base_length for i in range(n)]
    else:
        split_lens = split_cnt
    split_lens = torch.tensor(split_lens, dtype=torch.int32)

    assert sum(split_lens) == seq_len
    split_seqs = sequence.split(split_lens, dim=-1)
    return split_seqs, split_lens


def tree_partition_pipeline(draft_tokens, tree_position_ids, tree_attention_mask, retrieve_indices, total_stage: int):
    """
    split the input tree into multiple subtrees
    :param total_stage:
    :param draft_tokens: flattened tree token sequence following the causal rule
    :param tree_position_ids:
    :param tree_attention_mask:
    :param retrieve_indices:
    :return: Tuple(subtree1=(draft_tokens, tree_position_ids, tree_attention_mask, retrieve_indices),...)
    """
    seqs_split, lens_split = split_sequence_close_equal_len(draft_tokens, total_stage)
    tree_pos_ids_split = tree_position_ids.split(lens_split)
    # cum_seq_lens = torch.cumsum(lens_split, dim=-1)
    return zip(seqs_split, tree_pos_ids_split, lens_split)


def stage_tree_decoding(
        stage_model: StageEaModel,
        stage_past_key_values=None,
        draft_seqs_split=None,
        lens_split=None,
        tree_pos_ids_split=None,
        tree_attention_mask=None,
        input_ids=None,
        retrieve_indices=None,
):
    """
    pipelined tree decoding for verification
    :param stage_model: necessary
    :param stage_past_key_values: necessary
    :param draft_seqs_split: only necessary for stage1
    :param lens_split: only necessary for stage1
    :param tree_pos_ids_split: only necessary for stage1
    :param tree_attention_mask: only necessary for stage1
    :param input_ids: only necessary for stage1
    :param retrieve_indices: only necessary for stage(-1)
    :return: return result on stage1
    """
    # [step1] prepare necessary data to all devices
    # todo: overlap with the computation of the 1st subseq on the 1st stage
    config = stage_model.config
    if stage_model.is_first_stage:
        assert None not in (draft_seqs_split, tree_pos_ids_split, tree_attention_mask, input_ids, retrieve_indices)
        mask_shape = torch.tensor(tree_attention_mask.shape, dtype=torch.int32)
        dist.broadcast(mask_shape, src=0)
        dist.broadcast(lens_split)
        dist.send(retrieve_indices, dst=config.last_rank)
    else:
        # send shape first
        mask_shape = torch.zeros(4, dtype=torch.int32)  # shape like (1, 1, seq_len, seq_len)
        dist.broadcast(mask_shape, src=0)
        lens_split = torch.zeros(stage_model.total_stage, dtype=torch.int32)
        dist.broadcast(lens_split, src=0)
        tree_attention_mask = torch.zeros(*mask_shape, dtype=torch.float32)  # dtype of mask maybe defined in a config
        stage_model.tree_attention_mask = tree_attention_mask  # set the attn_mask in stage model
    if stage_model.is_last_stage:
        tree_logits_split = ()
    dist.broadcast(tree_attention_mask, src=0)
    cum_lens_split = torch.cumsum(lens_split, dim=-1)
    # [step1] end

    # [step2] start pipelined verification
    isend_task = None
    # irecv_task = None
    for i, subseq_len in enumerate(lens_split):  # todo: what happen when n_subseq > n_stage
        if stage_model.is_first_stage:
            if isend_task is not None:
                isend_task.wait()  # todo: set the timeout in config
            seq_split = draft_seqs_split[i]
            pos_ids_split = tree_pos_ids_split[i]
            hidden_state = stage_model(
                input_ids=seq_split,
                past_key_values=stage_past_key_values,
                position_ids=pos_ids_split,
                tree_mask_range=cum_lens_split[i]  # todo: get partial tree mask in forward of stage model
            )
            isend_task = dist.isend(hidden_state, dst=config.next_stage)

        else:
            last_hidden_state = torch.zeros((1, subseq_len, config.d_model), dtype=torch.float16)  # todo: d_model?
            dist.recv(last_hidden_state, src=config.last_rank)  # todo: need irecv?

            if not stage_model.is_last_stage:
                # middle stage
                hidden_state = stage_model(
                    last_hidden_state=last_hidden_state,
                    past_key_values=stage_past_key_values,
                    tree_mask_range=cum_lens_split[i],
                )
                if isend_task is not None:
                    isend_task.wait()
                isend_task = dist.isend(hidden_state, dst=config.next_rank)
            else:
                # last stage
                subseq_outputs, subseq_tree_logits, hidden_state = stage_model(
                    last_hidden_state=last_hidden_state,
                    past_key_values=stage_past_key_values,
                    tree_mask_range=cum_lens_split[i],
                    output_orig=True,
                )
                tree_logits_split = tree_logits_split + (hidden_state, )
    # [step2] end

    # [step3] get complete tree_logits
    if stage_model.is_last_stage:
        tree_logits = torch.concat(tree_logits_split, dim=-2)  # concat at the seq dimension
        logits = tree_logits[0, retrieve_indices]



if __name__ == '__main__':
    sequence = torch.randint(0, 10000, (1, 209), dtype=torch.int32)
    res = split_sequence_close_equal_len(sequence, 6)
    for i in res:
        print(i)
