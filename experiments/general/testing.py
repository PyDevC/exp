import torch.nn as nn
import torch

import torch.func as fn
import torch.autograd as ao

class MultiOutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 3)
        self.layer3 = nn.Linear(5, 2)

    def forward(self, x):
        h = self.layer1(x)
        out1 = self.layer2(h)
        out2 = self.layer3(h)
        return out1, out2

model = MultiOutputModule()

t = torch.randn(10)
x = torch.randn(10)
def test_cos(params, t):
    call_model, params = fn.functional_call(model, params, t)
    return call_model

fn.grad(test_cos)(dict(model.named_parameters()), t)
