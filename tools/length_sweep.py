import torch
import gc
from profiler.profiler import prof

def length_sweep(stage_base_model):
    prompt_lengths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    latency = []
    device = stage_base_model.model.layers[0].self_attn.q_proj.weight.device
    stage_base_model.eval()
    with torch.no_grad():
        for i, prompt_length in enumerate(prompt_lengths):
            print(f"prompt_length: {prompt_length}")
            input_ids = torch.randint(0, stage_base_model.config.vocab_size - 200, (1, prompt_length)).to(device)
            # warm up
            for _ in range(5):
                outputs = stage_base_model(input_ids)
            # measure
            for _ in range(10):
                with prof.time_context(f"optimal length {prompt_length}"):
                    stage_base_model(input_ids)
            latency.append(prof.elapsed_time(f"optimal length {prompt_length}")[1])
            prof.delete_time_events(f"optimal length {prompt_length}")
            del input_ids
            gc.collect()
            torch.cuda.empty_cache()
    # print(latency)
    for i, l in enumerate(latency):
        if latency[i] < latency[i+1] * 0.95:
            return prompt_lengths[i]
    raise ValueError("No optimal length found")

if __name__ == "__main__":
    from config.run_config import config as run_config
    from stage_ea_model import StageEaModel
    
    stage_model = StageEaModel.from_pretrained(
                stage_base_model_path=run_config.base_model_dir + f"/stage_model_0",
                ea_model_path=run_config.EAGLE_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # max_memory={"cpu": "1GB"},
                use_safetensors=True,
                device_map=f"cuda:0",
                # # total_token=-1
            )
    print(length_sweep(stage_model.stage_base_model))
