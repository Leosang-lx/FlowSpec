import torch
from profiler.profiler import prof

def length_sweep(stage_base_model):
    prompt_lengths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    latency = []
    device = stage_base_model.model.layers[0].self_attn.q_proj.weight.device
    stage_base_model.eval()
    with torch.no_grad():
        for i, prompt_length in enumerate(prompt_lengths):
            input_ids = torch.randint(0, stage_base_model.config.vocab_size - 200, (1, prompt_length)).to(device)
            # warm up
            for _ in range(20):
                outputs = stage_base_model(input_ids)
            # measure
            for _ in range(20):
                with prof.time_context(f"optimal length {prompt_length}"):
                    outputs = stage_base_model(input_ids)
            latency.append(prof.elapsed_time(f"optimal length {prompt_length}")[1])
            prof.delete_time_events(f"optimal length {prompt_length}")
    print(latency)
    for i, l in enumerate(latency):
        if latency[i] < latency[i+1] * 0.90:
            return prompt_lengths[i]
    raise ValueError("No optimal length found")