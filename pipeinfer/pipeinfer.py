# speculative_llama.py

import os
import time
import math
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Dict, Any, Optional

# -------------------------
# —— 数据结构定义 —— 
# -------------------------
class SeqDraft:
    def __init__(self, ctx_sampling):
        self.active: bool = False
        self.drafting: bool = False
        self.skip: bool = False
        self.i_batch_dft: int = 0
        self.i_batch_tgt: List[int] = []
        self.tokens: List[int] = []
        self.prefix_tokens: List[int] = []
        self.ctx_sampling = ctx_sampling  # e.g. past_key_values copy

class SeqAsyncRun:
    def __init__(self):
        self.batch_input_ids: torch.Tensor       # [batch_size, seq_len]
        self.batch_attention_mask: torch.Tensor  # same shape
        self.drafts: List[SeqDraft]              # 完整快照
        self.n_past_tgt: int
        self.prefix_n_past_tgt: int
        self.n_past_dft: int
        self.n_past_max: int
        self.s_keep: int
        self.seq_offset: int
        self.speculative: bool
        self.canceled: bool

# -------------------------
# —— 参数和调整函数 —— 
# -------------------------
class Params:
    def __init__(self):
        self.n_parallel = 4       # 并行分支数
        self.n_draft = 8          # 每次从 draft 模型采样多少 token
        self.p_accept = 0.7
        self.p_split  = 0.9
        self.p_decay  = 0.01
        self.p_recovery = 0.005
        self.max_predict = 128

def calc_p_adjust(params: Params, itr: int, n_reject: int) -> float:
    return itr * params.p_recovery - max(n_reject * params.p_decay, 0.0)

# -------------------------
# —— 分布式同步辅助 —— 
# -------------------------
def sync_bool(flag: bool, src: int = 0) -> bool:
    t = torch.tensor(int(flag), device="cpu")
    dist.broadcast(t, src=src)
    return bool(t.item())

def sync_tensor(tensor: torch.Tensor, src: int = 0):
    dist.broadcast(tensor, src=src)
    return tensor

# -------------------------
# —— KV‐Cache 管理 —— 
# transformers 中，我们用 `past_key_values` 来管理
# -------------------------
def copy_past(past: List[tuple]) -> List[tuple]:
    # 深拷贝 past_key_values
    return [(k.clone(), v.clone()) for k,v in past]

def rm_past_slot(past: List[List[tuple]], slot: int):
    # 删除某个分支 slot 的 past
    for i, layer in enumerate(past):
        del layer[slot]

def cp_past_slot(src_past: List[List[tuple]], dst_past: List[List[tuple]], src_slot: int, dst_slot: int):
    # 复制 src_slot -> dst_slot
    for layer_idx in range(len(src_past)):
        dst_past[layer_idx][dst_slot] = (
            src_past[layer_idx][src_slot][0].clone(),
            src_past[layer_idx][src_slot][1].clone()
        )

# -------------------------
# —— 启动非推测异步解码 —— 
# -------------------------
def begin_non_spec_run(params: Params,
                       model: LlamaForCausalLM, tokenizer: LlamaTokenizer,
                       input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       drafts: List[SeqDraft],
                       generated_len: int,
                       queue: List[SeqAsyncRun],
                       device: torch.device):
    run = SeqAsyncRun()
    # batch 就是最后接受的那个 token
    run.batch_input_ids   = input_ids[:, generated_len-1 : generated_len].clone()
    run.batch_attention_mask = attention_mask[:, generated_len-1 : generated_len].clone()
    run.drafts = [SeqDraft(copy_past(d.ctx_sampling)) for d in drafts]
    run.n_past_tgt = generated_len
    run.n_past_dft = generated_len
    run.prefix_n_past_tgt = generated_len - 1
    run.n_past_max = generated_len
    run.seq_offset = 0        # 单卡场景可省略 slot 管理
    run.speculative = False
    run.canceled = False
    queue.append(run)

# -------------------------
# —— 启动推测异步解码 —— 
# -------------------------
def start_async_spec_run(params: Params,
                         model_dft: LlamaForCausalLM,
                         run: SeqAsyncRun,
                         drafts: List[SeqDraft],
                         device: torch.device) -> bool:
    """
    对每条活跃分支：复制 past_key_values，
    然后基于 p_accept / p_split 做 tree‐sampling，
    最后将结果装进 run 并返回。
    """
    # ... 由于篇幅，此处省略完整树分支逻辑，
    #       关键是：
    #   1) 拷贝 run.batch_input_ids 作为草稿模型的新输入
    #   2) 执行多步 greedy/sample，每步:
    #         outputs = model_dft(input_ids, attention_mask, past_key_values=...)
    #         probs = softmax(outputs.logits[:, -1, :])
    #         if probs[best] < p_accept: stop this branch
    #         else accept、可能 split 分支
    #   3) 更新每个 SeqDraft.tokens / ctx_sampling
    return False

# -------------------------
# —— 检查并取消不再匹配的 speculative run —— 
# -------------------------
def check_for_cancel(generated: List[int],
                     queue: List[SeqAsyncRun]) -> None:
    canceled = []
    for run in queue:
        if not run.speculative or run.canceled:
            continue
        # 从 run.prefix_n_past_tgt 开始，将 run.drafts 中每个分支的
        # prefix_tokens + tokens 和 generated 做对比，不匹配则取消
        match = False
        for d in run.drafts:
            seq = d.prefix_tokens + d.tokens
            if generated[run.prefix_n_past_tgt: run.prefix_n_past_tgt + len(seq)] == seq:
                match = True
                break
        if not match:
            run.canceled = True
            canceled.append(run)
    for c in canceled:
        queue.remove(c)

# -------------------------
# —— 推测循环入口 —— 
# -------------------------
def run_speculation_loop(params: Params,
                         model_tgt: LlamaForCausalLM,
                         model_dft: LlamaForCausalLM,
                         tokenizer: LlamaTokenizer,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         generated: List[int],
                         drafts: List[SeqDraft],
                         tgt_queue: List[SeqAsyncRun],
                         dft_queue: List[SeqAsyncRun],
                         device: torch.device):
    itr = 0
    n_reject = 0
    while True:
        p_adj = calc_p_adjust(params, itr, n_reject)
        if params.p_accept + p_adj >= 1.0:
            break
        # 启动一个新的 speculative run
        run = SeqAsyncRun()
        # copy current generated context into run.batch_input_ids
        run.batch_input_ids   = input_ids.clone()
        run.batch_attention_mask = attention_mask.clone()
        run.drafts = [SeqDraft(copy_past(d.ctx_sampling)) for d in drafts]
        run.n_past_tgt = len(generated)
        run.prefix_n_past_tgt = len(generated)
        run.n_past_dft = len(generated)
        run.n_past_max = len(generated)
        run.speculative = True
        run.canceled = False

        # 真正开始 draft
        stop = start_async_spec_run(params, model_dft, run, drafts, device)
        if stop:
            break
        tgt_queue.append(run)
        itr += 1

# -------------------------
# —— 主函数 —— 
# -------------------------
def main_worker(rank: int, world_size: int, params: Params):
    # —— 初始化分布式 —— 
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-7b")
    model_tgt = LlamaForCausalLM.from_pretrained("hf-internal-testing/llama-7b", torch_dtype=torch.float16).to(device)
    model_dft = LlamaForCausalLM.from_pretrained("hf-internal-testing/llama-7b", torch_dtype=torch.float16).to(device)

    prompt = "Hello, this is a test of speculative decoding."
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    # —— 前向 prompt —— 
    with torch.no_grad():
        out_t = model_tgt(**enc, use_cache=True)
        past_tgt = out_t.past_key_values
        out_d = model_dft(**enc, use_cache=True)
        past_dft = out_d.past_key_values

    generated = input_ids[0].tolist()
    n_input = len(generated)

    # 初始化草稿分支状态
    drafts = []
    for _ in range(params.n_parallel):
        d = SeqDraft(copy_past(past_dft))
        d.active = True
        d.drafting = True
        d.prefix_tokens = []
        d.tokens = []
        drafts.append(d)

    tgt_queue: List[SeqAsyncRun] = []
    dft_queue: List[SeqAsyncRun] = []

    n_predict = 0
    has_eos = False

    # —— 主生成循环 —— 
    while n_predict < params.max_predict and not has_eos:
        # 1) 检查并合并完成的 run（略）

        # 2) 推测循环
        run_speculation_loop(params, model_tgt, model_dft, tokenizer,
                             input_ids, attention_mask, generated,
                             drafts, tgt_queue, dft_queue, device)

        # 3) 从 target 模型采样一个 token
        with torch.no_grad():
            outputs = model_tgt(input_ids=input_ids, attention_mask=attention_mask,
                                past_key_values=past_tgt, use_cache=True)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]
        next_id = torch.multinomial(probs, num_samples=1).item()

        # 4) 接受 token
        generated.append(next_id)
        n_predict += 1
        if next_id == tokenizer.eos_token_id:
            has_eos = True

        # 5) 更新 past_key_values
        input_ids = torch.tensor([[next_id]], device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        past_tgt = outputs.past_key_values

        # 6) 匹配并剔除不匹配的 draft
        #    drafts[s].tokens 中保存了 speculative run 中的 tokens
        #    只保留那些与 next_id 匹配的 branches

        for d in drafts:
            if not d.active: continue
            if len(d.tokens)==0 or d.tokens[0]!= next_id:
                d.active = False
            else:
                # 移除已匹配 token
                d.tokens.pop(0)
                d.prefix_tokens.append(next_id)

        # 7) 启动非推测 fallback run，以保持 cache 同步
        begin_non_spec_run(params, model_tgt, tokenizer,
                           input_ids, attention_mask,
                           drafts, len(generated),
                           tgt_queue, device)
        begin_non_spec_run(params, model_dft, tokenizer,
                           input_ids, attention_mask,
                           drafts, len(generated),
                           dft_queue, device)

        # 8) 清理过期 speculative runs
        check_for_cancel(generated, tgt_queue)

    # 打印结果
    print("Generated:", tokenizer.decode(generated))

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    params = Params()

    if world_size > 1:
        # 使用 torch.multiprocessing.spawn 启动多进程
        from torch.multiprocessing import spawn
        spawn(main_worker, args=(world_size, params,), nprocs=world_size)
    else:
        main_worker(0, 1, params)
