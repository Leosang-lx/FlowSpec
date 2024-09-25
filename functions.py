import torch
import torch.nn as nn
import numpy as np


def count_params(model: nn.Module):
    return np.sum((p.numel() for p in model.parameters()), dtype=np.int64)


def estimated_model_params(model: nn.Module, element_size=4, unit='MB'):
    num_params = count_params(model)

    total_memory_B = num_params * element_size
    if unit == 'MB':
        total_memory_unit = total_memory_B / (2 ** 20)
    else:
        pass

    print(f"Model has {num_params} parameters, occupying approximately {total_memory_unit:.2f} {unit} of memory.")

    return num_params


def estimated_model_params_config(model_config_params, element_size=4, share_weight=True):
    V, P, N, d_model, h, d_h, r = model_config_params
    num_params = (P + V) * d_model + N * (4 + 2 * r) * (d_model ** 2)
    # small parameters: weight on LayerNorm and bias on Linear and LayerNorm
    bias_and_layerNorm = N * (4 + 3 + 1 + r + 1)
    num_params += bias_and_layerNorm
    if not share_weight:
        num_params += V * d_model
    model_memory_bytes = num_params * element_size

    print(
        f"Model has {num_params} parameters, occupying approximately {model_memory_bytes / (2 ** 20):.2f} MB of memory.")
    return num_params


def estimated_KV_cache(model_config_params, batch_and_length, use_cache=True):
    V, P, N, d_model, h, d_h, r = model_config_params
    b, t = batch_and_length
    if use_cache:
        KV_cache_size = np.prod([2, b, t, d_model, N], dtype=np.int64)
    else:
        KV_cache_size = np.prod([2, b, t, d_model], dtype=np.int64)
    # print(f"KV cache occupies approximately {KV_cache_size / (2 ** 18):.2f} MB of memory.")
    return KV_cache_size


from GPT2_autoregressive_inference import model, config, model_config

if __name__ == '__main__':
    estimated_model_params(model)
    estimated_model_params_config(model_config)
    estimated_KV_cache(model_config, (1, 1000))
    estimated_KV_cache(model_config, (100, 1000))
