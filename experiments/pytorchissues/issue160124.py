import torch
from torch.fx import symbolic_trace

def simple_sum_reduction(x):
    return torch.sum(x, dim=[0], keepdim=True, dtype=torch.float32)

input_tensor = torch.randn(1792, 3200, dtype=torch.bfloat16, device="cuda")
simple_sum_reduction(input_tensor)
traced = symbolic_trace(simple_sum_reduction, concrete_args={"x": input_tensor})
traced.print_readable()
graph = traced.graph
graph.print_tabular()
