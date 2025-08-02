import torch
import torch.nn as nn
import torch.distributed as dist
import math

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, tp_rank=0, tp_size=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        # print(f'out_features: {out_features}, tp_size: {tp_size}')
        # self.local_out_features = out_features // tp_size
        self.local_out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        self.bias = nn.Parameter(torch.empty(self.local_out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, tp_group=None):
        # Local linear
        assert input.shape[-1] == self.in_features, f'input.shape[-1]: {input.shape[-1]}, self.in_features: {self.in_features}'
        output = torch.matmul(input, self.weight.t())
        # print(f'column output.shape: {output.shape}')
        if self.bias is not None:
            output += self.bias

        # Gather across ranks
        # outputs = [torch.zeros_like(output) for _ in range(self.tp_size)]
        # dist.all_gather(outputs, output, group=tp_group)
        # return torch.cat(outputs, dim=-1)
        return output


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, tp_rank=0, tp_size=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"
        # self.local_in_features = in_features // tp_size
        self.local_in_features = in_features

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, tp_group=None):
        # input: [B, T, hidden]
        # Slice local input

        # Local matmul
        output = torch.matmul(input, self.weight.t())
        # print(f'row output.shape: {output.shape}')
        # Reduce across ranks
        # dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
        # if self.bias is not None:
        #     output += self.bias
        # return output
        return output
