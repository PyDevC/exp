import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        y = torch.Tensor([0])  # y dtype: torch.float32
        x = torch.slice_scatter(y, x, 0)
        return x


model = Model()

x = torch.Tensor([0]).to(torch.int64)

inputs = [x]


def run_test(model, inputs, backend):
    model.eval()
    torch.manual_seed(0)
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    try:
        c_output = model(*inputs)
        print(c_output)
    except Exception as e:
        print(e)


run_test(model, inputs, 'eager')
run_test(model, inputs, 'inductor')
