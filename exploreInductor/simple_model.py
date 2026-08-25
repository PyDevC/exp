import torch
from torch.fx import symbolic_trace
from torch._inductor import compile

class ModelNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

model = ModelNet()
gm = symbolic_trace(model)

compile(gm, example_inputs=[
    torch.randn(4, 4), torch.randn(1, 4),
    torch.randn(4, 2), torch.randn(2, 4)
])
