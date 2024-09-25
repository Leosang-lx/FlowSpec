from matplotlib import pyplot as plt
from functions import estimated_model_params, model, estimated_KV_cache
import seaborn as sns
import numpy as np
from GPT2_autoregressive_inference import model_config

if __name__ == '__main__':
    model_params_num = estimated_model_params(model)
    b_max = 256
    t_max = 1024
    data_map = np.full((b_max, t_max), fill_value=np.nan)
    for b in range(1, 1+b_max):
        first_over_model = True
        for t in range(1, 1+t_max):
            b_t = (b, t)
            num_KV_cache = estimated_KV_cache(model_config, b_t)
            if t == 1 and num_KV_cache <= model_params_num:
                first_over_model = False
            if first_over_model is False and num_KV_cache <= model_params_num:
                # try to draw a boundary about of the comparison of KV cache and model size
                data_map[b - 1, t - 1] = 50
                first_over_model = True
            else:
                data_map[b - 1, t - 1] = num_KV_cache / model_params_num
    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(data_map, cmap='coolwarm')
    ax.set_xticks([i if i == 0 else i-1 for i in range(0, t_max+1, 128)])
    ax.set_yticks([i if i == 0 else i-1 for i in range(0, b_max+1, 64)])
    ax.set_xticklabels([i+1 if i == 0 else i for i in range(0, t_max+1, 128)], rotation=0)
    ax.set_yticklabels([i+1 if i == 0 else i for i in range(0, b_max+1, 64)], rotation=0)
    ax.invert_yaxis()
    plt.ylabel('batch size')
    plt.xlabel('text length')
    plt.title('Ratio of $Params_{model} / Params_{KV}$ of different $(b,t)$')
    plt.tight_layout()

    plt.show()