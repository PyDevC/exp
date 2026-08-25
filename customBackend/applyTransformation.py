import torch
from torch.fx import symbolic_trace

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(x)

def replaceReluWithLinear(gm: torch.fx.GraphModule):
    linear = torch.nn.Linear(100, 20)
    gm.add_submodule('Linear', linear)
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            if node.target == 'relu':
                with gm.graph.inserting_after(node):
                    new_node = gm.graph.call_module('Linear', node.args)
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm

model = Net()
gm = symbolic_trace(model)
out = replaceReluWithLinear(gm)
out.graph.print_tabular()
