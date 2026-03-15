import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True)
def model_forward(x, weights=None):
    breakpoint()
    x = F.linear(x, weights)
    return x

weights = torch.tensor([2, 1])
x = torch.tensor([1, 2])
model_forward(x, weights)
