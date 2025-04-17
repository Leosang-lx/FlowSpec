from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from fastchat.model import get_conversation_template
from datetime import timedelta
from pipeline_utils import calculate_model_size_with_buffers
import torch
import torch.distributed as dist
import time
import argparse
import os
import warnings
from torch.profiler import ProfilerActivity
from profiler.profiler import prof
from config.run_config import config as run_config
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# torch.manual_seed(1234)

# os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '120'

def main(args):
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    # dist.init_process_group(backend='gloo', init_method='env://', timeout=timedelta(seconds=60))
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # device = rank % torch.cuda.device_count()
    device = 0
    torch.cuda.set_device(device)
    print(f'rank={rank}, world_size={world_size}, device={device}')
    
    # base_model_path = f"/home/liux/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/stage_model_series_8+8+8+8/stage_model_{rank}"
    # base_model_path = f"/home/nvidia/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/stage_model_series_6+9+9+8_half/stage_model_{rank}"
    # EAGLE_model_path = "/home/nvidia/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    
    if rank == 0:
        prof.time_start("total_time")
    with prof.profile_context(f"Rank {rank}: loading stage model", device=f"cuda:{device}"):
        if rank == 0:
            stage_model = StageEaModel.from_pretrained(
                stage_base_model_path=run_config.base_model_dir + f"/stage_model_{rank}",
                ea_model_path=run_config.EAGLE_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # max_memory={"cpu": "1GB"},
                use_safetensors=True,
                device_map=f"cuda:{device}",
                # # total_token=-1,
                total_token=run_config.total_token,
                depth=run_config.depth,
            )
        else:
            stage_model = StageEaModel.from_pretrained(
                stage_base_model_path=run_config.base_model_dir + f"/stage_model_{rank}",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # max_memory={"cpu": "1GB"},
                use_safetensors=True,
                device_map=f"cuda:{device}",
                total_token=run_config.total_token,
                depth=run_config.depth,
            )
    
    # stage_model.to(f"cuda:{device}")
    # stage_model.stage_base_model.to(f"cuda:{device}")
    # if rank == 0:
    #     stage_model.ea_layer.to(f"cuda:{device}")
    #     stage_model.ea_layer.embed_tokens.to(f"cuda:{device}")
    stage_model.eval()

    with torch.no_grad():
        assert run_config.pipeline_type in ["naive", "pruned", "continuous"]
        if rank == 0:
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

            input_ids=stage_model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()
            
            # start = time.perf_counter()
            # log = True
            
            with prof.profile_context(f"Rank {rank}: eagenerate", device=f"cuda:{device}"):
                outputs = stage_model.stage_generate(input_ids, temperature=run_config.temperature, max_new_tokens=run_config.max_new_tokens, log=run_config.log)
                # if run_config.pipeline_type == "naive":
                #     outputs = stage_model.eagenerate_pipeline(input_ids,temperature=run_config.temperature,max_new_tokens=run_config.max_new_tokens, log=run_config.log)
                # elif run_config.pipeline_type == "pruned":
                #     outputs = stage_model.eagenerate_pruned_pipeline(input_ids, temperature=run_config.temperature, max_new_tokens=run_config.max_new_tokens, log=run_config.log)
                # elif run_config.pipeline_type == "continuous":
                #     outputs = stage_model.eagenerate_continuous(input_ids, temperature=run_config.temperature, max_new_tokens=run_config.max_new_tokens, log=run_config.log)
            if run_config.log:
                if len(outputs) == 3:
                    output_ids, new_tokens, idx = outputs
                else:
                    output_ids, new_tokens, idx, turns = outputs
            else:
                output_ids = outputs
            # torch.cuda.synchronize()
            # end = time.perf_counter()

            output = stage_model.tokenizer.decode(output_ids[0])
            print('\n=========OUTPUT=========')
            print(output)

            if run_config.log:
                print('New tokens:', new_tokens)
                print('Rounds:', idx+1)
                if len(outputs) == 4:
                    print('Turns:', turns)
        else:
            with prof.profile_context(f"Rank {rank}: eagenerate", device=f"cuda:{device}"):
                stage_model.stage_generate(temperature=run_config.temperature, max_new_tokens=run_config.max_new_tokens)
                # if run_config.pipeline_type == "naive":
                #     stage_model.eagenerate_pipeline(temperature=run_config.temperature, max_new_tokens=run_config.max_new_tokens)
                # elif run_config.pipeline_type == "pruned":
                #     stage_model.eagenerate_pruned_pipeline(temperature=run_config.temperature, max_new_tokens=run_config.max_new_tokens)
                # elif run_config.pipeline_type == "continuous":
                #     stage_model.eagenerate_continuous(temperature=run_config.temperature, max_new_tokens=run_config.max_new_tokens)
    
    if rank == 0:
        print(torch.cuda.list_gpu_processes(device=f"cuda:{device}"))
        prof.time_stop("total_time")
    
    dist.barrier() # let output looks neat
    prof.print_all_events()
    
    dist.barrier()
    stage_model.comm.stop()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = args.parse_args()
    main(args)