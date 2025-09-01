import torch
import torch.nn as nn


class CondModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cond(x.ndim, lambda x_: x_.mean(), lambda x_: x_, x)


if __name__ == "__main__":
    model = CondModel()
    g = torch.fx.symbolic_trace(model)

