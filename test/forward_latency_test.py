from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from fastchat.model import get_conversation_template
from fastchat.llm_judge.common import load_questions
# from datetime import timedelta
from pipeline_utils import calculate_model_size_with_buffers, get_time_str
import torch
from tqdm import tqdm
# import time
import argparse
import os
import warnings
# from torch.profiler import ProfilerActivity
from profiler.profiler import prof, is_strictly_ascending, save_as
from contextlib import nullcontext
from config.run_config import config as run_config
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")
import time

rank = 1
assert torch.cuda.is_available()
torch.set_grad_enabled(False)

# rank = int(os.environ['RANK'])
# device = rank % torch.cuda.device_count()
device = 0
torch.cuda.set_device(device)
# print(f'rank={rank}, world_size={world_size}, device={device}')
print(run_config.EAGLE_model_path)
# with prof.profile_context(f"Rank {rank}: loading stage model", device=f"cuda:{device}"):
stage_model = StageEaModel.from_pretrained(
    stage_base_model_path=run_config.base_model_dir + f"/stage_model_{rank}",
    ea_model_path=run_config.EAGLE_model_path if rank == 0 else None,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    # max_memory={"cpu": "1GB"},
    init_comm=False,
    use_safetensors=True,
    quantization_config=run_config.quant_config,
    device_map=f"cuda:{device}",
    # total_token=run_config.init_total_token,
    # depth=run_config.init_depth,
    # top_k=run_config.init_topk if run_config.pipeline_type != "pipedec" else run_config.init_topk_pipedec,
)

# # check shared_weight
# # if stage_model.config.has_lm_head:
# #     assert stage_model.stage_base_model.lm_head.weight is stage_model.stage_base_model.model.embed_tokens.weight

# stage_model.eval()

warmup_cnt = 20

@torch.no_grad()
def test_forward_latency(x, use_cache=False, past_key_values=None):
    # input_ids = torch.zeros(1, input_length).long().cuda()
    torch.cuda.synchronize()
    start_time = time.time()

    _ = stage_model(
        input_ids=x,
        past_key_values=past_key_values,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def warmup(warmup_cnt, func, x):
    print(f"Warmup {warmup_cnt} times...")
    for i in tqdm(range(warmup_cnt)):
        func(x)

def eval_hidden_layers():
    print('hidden_layers')
    warmup(warmup_cnt, stage_model, torch.zeros(1, 100).long().cuda())
        
    test_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for test_len in test_lens:
        print(f"Test Input length: {test_len}")
        avg = 0
        dummpy_input = torch.zeros(1, test_len).long().cuda()
        for i in tqdm(range(10)):
            latency = test_forward_latency(dummpy_input)
            avg += latency
        avg /= 10
        print(f"Input length: {test_len}, Average latency: {avg} seconds")

@torch.no_grad()
def test_lm_head(lm_head, hidden_state):
    # dim = stage_model.config.vocab_size
    # hidden_state = torch.randn(1, 1, dim)
    torch.cuda.synchronize()
    start_time = time.time()
    
    _ = lm_head(hidden_state)
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

def eval_lm_head(vocab_size=32000, hidden_size=4096):
    lm_head = torch.nn.Linear(hidden_size, vocab_size).to(torch.float16).cuda()
    print(f'lm_head size: {lm_head.weight.size()}')
    warmup(warmup_cnt, lm_head, torch.randn(1, 16, hidden_size, dtype=torch.float16).cuda())

    test_lens = [1, 2, 4, 8, 16, 32, 64]
    for test_len in test_lens:
        print(f"Test Input length: {test_len}")
        avg = 0
        dummpy_input = torch.randn(1, test_len, hidden_size, dtype=torch.float16).cuda()
        for i in tqdm(range(10)):
            latency = test_lm_head(lm_head, dummpy_input)
            avg += latency
        avg /= 10
        print(f"Input length: {test_len}, Average latency: {avg*1000: .3f} ms")


@torch.no_grad()
def test_draft_model(top_k=10, depth=6, total_token=64):
    torch.cuda.synchronize()
    start_time = time.time()
    _ = stage_model.ea_layer.topK_genrate()
    torch.cuda.synchronize()



            
if __name__ == "__main__":
    print('Eval forward latency')
    # eval_hidden_layers()

    eval_lm_head(128256)
