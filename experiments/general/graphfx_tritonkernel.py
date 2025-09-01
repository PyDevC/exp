import torch.nn as nn
import torch
from torch.fx import symbolic_trace

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.linear = nn.Linear(100,100)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x: torch.Tensor):
        self.linear(x)
        self.relu(x)
        self.linear(x)
        return self.softmax(x)

t = torch.randint(1,10, size=(100,))
model = net()

traced = symbolic_trace(model)
traced.print_readable()
graph = traced.graph
graph.print_tabular()
