import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def split_LN(split_embedding, x_mean, x_var, split_w, split_b, eps):
    # split_embedding.sub_(x_mean).div_(torch.sqrt(x_var + eps))
    y = (split_embedding - x_mean) / torch.sqrt(x_var + eps)
    if split_b is not None and split_b is not None:
        return y * split_w + split_b
    return y

def test_LN():
    d_model = 256
    eps = 1e-3
    LN_layer = nn.LayerNorm(d_model, eps)
    LN_layer.eval()

    ln_w = LN_layer.weight
    ln_b = LN_layer.bias

    init.uniform_(ln_w)
    init.uniform_(ln_b)

    x = torch.randn(1, 2, d_model)
    # forward()
    y = LN_layer(x)
    # functional
    yf = F.layer_norm(x, (d_model,), ln_w, ln_b, eps=eps)
    print('Functional test [equal]:', torch.equal(y, yf))

    # split_LN
    split_cnt = 2
    split_embedding_sizes = [d_model // split_cnt] * split_cnt
    split_embeddings = x.split(split_embedding_sizes, dim=-1)
    split_ln_ws = ln_w.split(split_embedding_sizes, dim=-1)
    split_ln_bs = ln_b.split(split_embedding_sizes, dim=-1)
    split_ln_wb = [(split_ln_ws[i], split_ln_bs[i]) for i in range(split_cnt)]
    x_mean = x.mean(dim=-1, keepdim=True)
    x_var = x.var(dim=-1, keepdim=True, unbiased=False)

    x_mean_split = sum([se.sum(dim=-1, keepdim=True) for se in split_embeddings]).div_(d_model)
    x_var_split = sum([(se - x_mean).square().sum(dim=-1, keepdim=True) for se in split_embeddings]).div_(d_model)
    # split_embeddings = [F.layer_norm(se, (split_embedding_sizes[i],), *split_ln_wb[i], eps) for
    #                     i, se in enumerate(split_embeddings)]

    split_embeddings = [split_LN(se, x_mean, x_var, *split_ln_wb[i], eps) for i, se in enumerate(split_embeddings)]
    y_split = torch.concat(split_embeddings, dim=-1)
    print('Split test [equal]:', torch.equal(y, y_split))
    print('Split test [allclose]:', torch.allclose(y, y_split, atol=1e-3))
    a = 1


if __name__ == '__main__':
    test_LN()
