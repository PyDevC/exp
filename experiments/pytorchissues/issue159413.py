import torch


@torch.compile()
def foo():
    return torch.zeros(418119680000, dtype=torch.float32, device='cuda')


foo()
