import torch
import torch._dynamo as dynamo

def get_magic_number():
    dynamo.graph_break()
    return 42

@torch.compile(fullgraph=True)
def func(x):
    n = dynamo.nonstrict_trace(get_magic_number)()
    return x + n

func(torch.rand(20))
