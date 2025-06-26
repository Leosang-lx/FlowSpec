import os
import deepspeed
from deepspeed.inference.config import DeepSpeedInferenceConfig, DeepSpeedTPConfig
import torch
from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel
from eagle.config.run_config import config as run_config

def main():
    deepspeed.init_distributed(dist_backend='gloo', init_method='env://')  # :contentReference[oaicite:0]{index=0}

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)                                    # :contentReference[oaicite:1]{index=1}

    tokenizer = AutoTokenizer.from_pretrained(run_config.base_model_path)  # :contentReference[oaicite:2]{index=2}
    model = EaModel.from_pretrained(
        base_model_path=run_config.base_model_path,
        ea_model_path=run_config.eagle_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=f"cuda:{local_rank}",
        total_token=run_config.total_token,
        depth=run_config.depth,
    )
    model.eval()

    engine = deepspeed.init_inference(
        model=model,
        mp_size=2,                          # 张量并行度 :contentReference[oaicite:3]{index=3}
        dtype=torch.float16,                # 半精度 :contentReference[oaicite:4]{index=4}
        replace_method='auto',
        replace_with_kernel_inject=True,    # 注入高性能内核 :contentReference[oaicite:5]{index=5}
    )

    inputs = tokenizer.encode(run_config.run_text, return_tensors='pt').to(engine.device)
    outputs = engine.generate(inputs, max_new_tokens=10)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
