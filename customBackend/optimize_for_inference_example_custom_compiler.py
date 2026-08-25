from typing import Callable

import torch

def optimize(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    jit = torch.jit.script(gm)
    return torch.jit.optimize_for_inference(jit)


# Temp model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(x)

compiled_model = torch.compile(Net(), backend=optimize)
test = torch.Tensor([1, 2, 3, -1, -2, -3])
out = compiled_model(test)
print(out)
