from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from fastchat.model import get_conversation_template
from datetime import timedelta
from pipeline_utils import calculate_model_size_with_buffers
import torch
import torch.distributed as dist
from tqdm import tqdm
import time
import argparse
import os
import warnings
from torch.profiler import ProfilerActivity
from profiler.profiler import prof
from contextlib import nullcontext
from config.run_config import config as run_config
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# torch.manual_seed(1234)
# os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '120'

def main(args):
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # device = rank % torch.cuda.device_count()
    device = 1
    torch.cuda.set_device(device)
    print(f'rank={rank}, world_size={world_size}, device={device}')
    
    if rank == 0 and run_config.mode == "demo":
        prof.time_start("total_time")
    # with prof.profile_context(f"Rank {rank}: loading stage model", device=f"cuda:{device}"):
    stage_model = StageEaModel.from_pretrained(
        stage_base_model_path=run_config.base_model_dir + f"/stage_model_{rank}",
        ea_model_path=run_config.EAGLE_model_path if rank == 0 else None,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # max_memory={"cpu": "1GB"},
        use_safetensors=True,
        device_map=f"cuda:{device}",
        total_token=run_config.init_total_token,
        depth=run_config.init_depth,
        top_k=run_config.init_topk,
    )

    # check shared_weight
    # if stage_model.config.has_lm_head:
    #     assert stage_model.stage_base_model.lm_head.weight is stage_model.stage_base_model.model.embed_tokens.weight
    
    stage_model.eval()

    assert run_config.pipeline_type in ["naive", "pruned", "continuous"]
    assert run_config.mode in ["eval", "demo"]

    # [initialize]
    if rank == 1:
        your_message=run_config.your_message
        # conv = get_conversation_template("vicuna")
        conv = get_conversation_template("llama-2-chat")

        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p

        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " "
        print('\n=========PROMPT=========')
        print(prompt)

        input_ids = stage_model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()
    
    # collaborative generation
    def run(log=False, profiler=None):
        outputs = stage_model.stage_generate(
            input_ids=input_ids if rank == 1 else None,
            temperature=run_config.temperature,
            max_new_tokens=run_config.max_new_tokens,
            log=log if rank == 0 else False,
            pipeline_type=run_config.pipeline_type,
            profiler=profiler,
        )
        if rank == 0:
            return outputs

    # [warm-up]
    if run_config.warmup:
        cnt = tqdm(range(run_config.warmup_repeat), desc="Warmup") if rank == 1 else range(run_config.warmup_repeat)
        for _ in cnt:
            run()

    # [test generation]
    cnt = tqdm(range(run_config.test_repeat), desc="Test") if rank == 1 else range(run_config.test_repeat)
    for i in cnt:
        with prof.profile_context(f"Rank {rank}: {run_config.pipeline_type} pipeline", device=f"cuda:{device}"):
            outputs = run(run_config.log, prof)
    
    # [print output]
    if rank == 0:  # only for greedy decoding test!!!
        if run_config.log:
            output_ids, new_tokens, idx, turns = outputs
        else:
            output_ids = outputs
        output = stage_model.tokenizer.decode(output_ids[0])
        print('\n=========OUTPUT=========')
        print(output)
        if run_config.log:
            print('New tokens:', new_tokens)
            print('Rounds:', idx+1)
            if len(outputs) == 4:
                print('Turns:', turns)
    
    dist.barrier()
    # if rank == 0 or rank == world_size - 1:
    prof.print_all_events()
    
    # dist.barrier()
    if hasattr(stage_model, "comm"):
        stage_model.comm.stop()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = args.parse_args()
    main(args)