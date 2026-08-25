from torch.fx import symbolic_trace
from testmodels.simple import SimpleNet

model = SimpleNet()

gm = symbolic_trace(model)
gm.graph.print_tabular()
