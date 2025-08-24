import torch
import torch._inductor as inductor

def simple_sum_reduction(x):
    return torch.sum(x, dim=[0], keepdim=True, dtype=torch.float32)

input_tensor = torch.randn(1792, 3200, dtype=torch.bfloat16, device="cuda")
simple_sum_reduction(input_tensor)
