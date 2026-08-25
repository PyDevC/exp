import torch
from torch._dynamo.backends.common import aot_autograd

from typing import Callable

def bite_compile(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    gm.graph.print_tabular()
    return gm

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(x)


bite_compile = aot_autograd(fw_compiler=bite_compile, bw_compiler=bite_compile)

compiled_model = torch.compile(Net(), backend=bite_compile)
test = torch.Tensor([1, 2, 3, -1, -2, -3])
out = compiled_model(test)
print(out)
