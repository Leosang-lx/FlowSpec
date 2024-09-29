import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def transformer_autoregressive_inference_parameters(n_0, n_e, N, d_model, h, d_h, r):
    """
    List all parameters concerning the FLOPs complexity of LLM inference
    :return: package the parameters
    """
    print(f'Sequence length: [initial length n_0={n_0}], [max length n_e={n_e}]')
    print(f'Model depth: [layer_num N={N}]')
    print(f'Model dimension: [hidden dimension d_model={d_model}]')
    print(f'Multi-head attention: [heads h={h}], [head dimension d_h={d_h}]')
    print(f'Feed-forward layer: [hidden rate r={r}]')
    print(f'(where d_model = h * d_h)')
    return n_0, n_e, N, d_model, h, d_h


# The following formulation of FLOPs ignore Softmax, element-wise division, LayerNorm and Dropout
def prefill_phase_FLOPs(model_config_params, text_length):
    V, P, N, d_model, h, d_h, r = model_config_params
    n_0, n_e = text_length
    return N * ((8 + 4 * r) * n_0 * (d_model ** 2) + 4 * (n_0 ** 2) * d_model)


def decoding_phase_FLOPs(model_config_params, text_length):
    V, P, N, d_model, h, d_h, r = model_config_params
    n_0, n_e = text_length
    return N * ((n_e - n_0 - 1) * (8 + 4 * r) * (d_model**2) + 2 * (n_0 + n_e) * (n_e - n_0 - 1) * d_model)


n_0, n_e = 100, 200
V, P = 21128, 1024
N, d_model = 12, 768
h, d_h = 12, 7
r = 4
model_config = (V, P, N, d_model, h, d_h, r)
text_length = (n_0, n_e)
max_length_limit = 1024


if __name__ == '__main__':
    inference_parameters = transformer_autoregressive_inference_parameters(n_0, n_e, N, d_model, h, d_h, r)
    # 创建一个空的数据矩阵
    data_matrix = np.full((max_length_limit, max_length_limit), fill_value=np.nan)

    # 遍历所有可能的 (n0, ne) 对
    for ne in range(2, max_length_limit + 1):
        n_e = ne
        for n0 in range(1, ne):
            n_0 = n0
            text_length = (n_0, n_e)

            # 计算两个方法的结果
            prefillFLOPs = prefill_phase_FLOPs(model_config, text_length)
            decodingFLOPs = decoding_phase_FLOPs(model_config, text_length)

            # 计算比值
            ratio = 1 - prefillFLOPs / (prefillFLOPs + decodingFLOPs)

            # 将比值存储在数据矩阵中
            data_matrix[ne - 1, n0 - 1] = ratio

    # 绘制热力图
    # data_matrix.transpose((0, 1))
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(data_matrix, cmap='coolwarm')
    ax.set_xticks([i if i == 0 else i - 1 for i in range(0, max_length_limit + 1, 128)])
    ax.set_yticks([i if i == 0 else i - 1 for i in range(0, max_length_limit + 1, 128)])
    ax.set_xticklabels([i + 1 if i == 0 else i for i in range(0, max_length_limit + 1, 128)], rotation=0)
    ax.set_yticklabels([i + 1 if i == 0 else i for i in range(0, max_length_limit + 1, 128)], rotation=0)
    # ax.invert_yaxis()
    plt.xlabel('$n_0$')
    plt.ylabel('$n_e$')
    plt.title('Ratio of Prefill & Decoding FLOPs for different $(n_0,n_e)$')
    plt.tight_layout()
    plt.show()