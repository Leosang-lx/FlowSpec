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
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# torch.manual_seed(1234)

# os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '120'

def main(args):
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    dist.init_process_group(backend='gloo', init_method='env://', timeout=timedelta(seconds=15))
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # device = rank % torch.cuda.device_count()
    device = 1
    torch.cuda.set_device(device)
    print(f'rank={rank}, world_size={world_size}, device={device}')
    
    base_model_path = f"/home/liux/LLM/pipeline_model/meta-llama/Llama-2-7b-chat-hf/stage_model_series_8+8+8+8/stage_model_{rank}"
    EAGLE_model_path = "/home/liux/LLM/models_hf/yuhuili/EAGLE-llama2-chat-7B"
    # print(f'base_model_path={base_model_path}, EAGLE_model_path={EAGLE_model_path}')
    if rank == 0:
        stage_model = StageEaModel.from_pretrained(
            stage_base_model_path=base_model_path,
            ea_model_path=EAGLE_model_path,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            device_map=f"cuda:{device}",
            # # total_token=-1,
            total_token=64,
            depth=6,
        )
    else:
        stage_model = StageEaModel.from_pretrained(
            stage_base_model_path=base_model_path,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            device_map=f"cuda:{device}",
            total_token=64,
            depth=6,
        )
    
    model_size = calculate_model_size_with_buffers(stage_model)
    print(f'Rank{rank} Model: {model_size:.2f} MB')

    stage_model.to(f"cuda:{device}")
    stage_model.stage_base_model.to(f"cuda:{device}")
    if rank == 0:
        stage_model.ea_layer.to(f"cuda:{device}")
        stage_model.ea_layer.embed_tokens.to(f"cuda:{device}")
    stage_model.eval()
    
    # for i in range(10):
    #     torch.manual_seed(12345+i)
    if rank == 0:
        your_message="Hello"
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
        
        start = time.perf_counter()
        log = True
        # outputs = stage_model.eagenerate_pipeline(input_ids,temperature=0.5,max_new_tokens=512, log=log)
        # outputs = stage_model.eagenerate_pruned_pipeline(input_ids, temperature=0.5, max_new_tokens=512, log=log)
        outputs = stage_model.eagenerate_continuous(input_ids, temperature=0.5, max_new_tokens=512, log=log)
        if log:
            output_ids, new_tokens, idx = outputs
        else:
            output_ids = outputs
        torch.cuda.synchronize()
        end = time.perf_counter()

        output = stage_model.tokenizer.decode(output_ids[0])
        print('\n=========OUTPUT=========')
        print(output)

        if log:
            print('New tokens:', new_tokens)
            print('Rounds:', idx+1)
        print(f'Total Inference time: {end - start:.2f}s')

    else:
        # stage_model.eagenerate_pipeline(temperature=0.5, max_new_tokens=512)
        # stage_model.eagenerate_pruned_pipeline(temperature=0.5, max_new_tokens=512)
        stage_model.eagenerate_continuous(temperature=0.5, max_new_tokens=512)
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = args.parse_args()
    main(args)