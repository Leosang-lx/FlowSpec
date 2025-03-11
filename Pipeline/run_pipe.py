from stage_ea_model import StageEaModel
from stage_ea_config import StageEaConfig
from fastchat.model import get_conversation_template
from datetime import timedelta
import torch
import torch.distributed as dist

import argparse

import os

# os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '120'

def main(args):
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    dist.init_process_group(backend='gloo', init_method='env://', timeout=timedelta(seconds=120))
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
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
            total_token=16,
            depth=2,
        )
    else:
        stage_model = StageEaModel.from_pretrained(
            stage_base_model_path=base_model_path,
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            device_map=f"cuda:{device}",
            total_token=16,
            depth=2,
        )
    stage_model.to(f"cuda:{device}")
    stage_model.stage_base_model.to(f"cuda:{device}")
    if rank == 0:
        stage_model.ea_layer.to(f"cuda:{device}")
        stage_model.ea_layer.embed_tokens.to(f"cuda:{device}")
    stage_model.eval()
    
    your_message="Hello"
    # conv = get_conversation_template("vicuna")
    conv = get_conversation_template("llama-2-chat")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if rank == 0:
        input_ids=stage_model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()
        print(f"input_ids: {input_ids}")
        output_ids=stage_model.eagenerate_pipeline(input_ids,temperature=0.5,max_new_tokens=512)
        output=stage_model.tokenizer.decode(output_ids[0])
        print(output)
    else:
        stage_model.eagenerate_pipeline(temperature=0.5, max_new_tokens=512)
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = args.parse_args()
    main(args)