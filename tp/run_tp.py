from tp_ea_model import TPEaModel
from stage_ea_config import StageEaConfig
from fastchat.model import get_conversation_template
from fastchat.llm_judge.common import load_questions
# from datetime import timedelta
from pipeline_utils import calculate_model_size_with_buffers, get_time_str
import torch
import torch.distributed as dist
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

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# torch.manual_seed(1234)
# os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '120'

def main():
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # device = rank % torch.cuda.device_count()
    device = 0
    torch.cuda.set_device(device)
    print(f'rank={rank}, world_size={world_size}, device={device}')
    
    # with prof.profile_context(f"Rank {rank}: loading stage model", device=f"cuda:{device}"):
    tp_model = TPEaModel.from_pretrained(
        tp_base_model_path= f"/home/liux/big_file/tp_model/meta-llama/Llama-2-7b-chat-hf/new_stage_model_series_tp_fp16/stage_model_{rank}",
        ea_model_path=run_config.EAGLE_model_path if rank == 0 else None,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # max_memory={"cpu": "1GB"},
        use_safetensors=True,
        quantization_config=run_config.quant_config,
        device_map=f"cuda:{device}",
        total_token=run_config.init_total_token,
        depth=run_config.init_depth,
        top_k=run_config.init_topk if run_config.pipeline_type != "pipedec" else run_config.init_topk_pipedec,
    )

    # state_dict = tp_model.state_dict()
    # if rank == 0:
    #     for key, value in state_dict.items():
    #         print(f'rank: {rank} key: {key} value: {value} shape: {value.shape}')
    # check shared_weight
    # if stage_model.config.has_lm_head:
    #     assert stage_model.stage_base_model.lm_head.weight is stage_model.stage_base_model.model.embed_tokens.weight
    
    tp_model.eval()
    # [update] pipedec
    assert run_config.pipeline_type in ["serial", "naive", "pruned", "continuous", "pipedec", "tp"]
    assert run_config.mode in ["eval", "demo"]

    # [initialize]
    if rank == 0:
        your_message=run_config.your_message
        if "vicuna" in run_config.model_name:
            conv = get_conversation_template("vicuna")
        elif "llama2" in run_config.model_name:
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
        elif "llama3" in run_config.model_name:
            messages = [
                {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            
        if "llama3" in run_config.model_name:
            messages.append({
                "role": "user",
                "content": your_message
            })
            prompt = tp_model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tp_model.tokenizer([prompt],add_special_tokens=False,).input_ids
        else:
            conv.append_message(conv.roles[0], your_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            print('\n=========PROMPT=========')
            print(prompt)

            input_ids = tp_model.tokenizer([prompt]).input_ids
        
        input_ids = torch.as_tensor(input_ids).cuda()

    # [warm-up]
    if run_config.warmup:
        cnt = tqdm(range(run_config.warmup_repeat), desc="Warmup") if rank == 0 else range(run_config.warmup_repeat)
        for _ in cnt:
            dist.barrier()
            outputs = run(
                    tp_model, 
                    input_ids if rank == 0 else None,
                    galaxy=run_config.use_galaxy,
                )

    # [test generation]
    cnt = tqdm(range(run_config.test_repeat), desc="Test") if rank == 0 else range(run_config.test_repeat)
    for i in cnt:
        dist.barrier()
        with prof.profile_context(f"Rank {rank}: {run_config.pipeline_type} pipeline", device=f"cuda:{device}") if run_config.prof else nullcontext():
            outputs = run(
                    tp_model, 
                    input_ids if rank == 0 else None, 
                    run_config.log if rank == 0 else False, 
                    prof if run_config.prof else None,
                    galaxy=run_config.use_galaxy,
                )
    
    # [print output]
    if rank == 0:  # only for greedy decoding test!!!
        if run_config.log:
            output_ids, new_tokens, idx, turns = outputs
        else:
            output_ids = outputs
        output = tp_model.tokenizer.decode(output_ids[0])
        print('\n=========OUTPUT=========')
        print(output)
        if run_config.log:
            print('New tokens:', new_tokens)
            print('Rounds:', idx+1)
            if len(outputs) == 4:
                print('Turns:', turns)

        # [update] for cumulative timing
        # print(prof.cumulative_time_events[0]['timestamp'])
        # print(prof.cumulative_time_events[0]['events'])
    
    dist.barrier()
    # if rank == 0 or rank == world_size - 1:
    if run_config.save_timestamps:
        assert is_strictly_ascending(prof.cumulative_time_events[0]['timestamp'])
        save_as(prof.cumulative_time_events, f'records/{get_time_str()}-rank{rank}-ws{world_size}.rec')

    prof.print_all_events()
    
    # dist.barrier()
    if hasattr(tp_model, "comm"):
        tp_model.comm.stop()
    dist.destroy_process_group()
    
    # reset traffic
    if run_config.hardware == "jetson":
        tp_model.comm.reset_traffic()
    

def run(tp_model, input_ids, log=False, profiler=None, galaxy=False):
    outputs = tp_model.tp_generate(
        input_ids=input_ids,
        temperature=run_config.temperature,
        max_new_tokens=run_config.max_new_tokens,
        log=log,
        profiler=profiler,
        galaxy=galaxy,
    )
    if dist.get_rank() == 0:
        return outputs
            
if __name__ == "__main__":
    assert run_config.mode == "demo"
    main()