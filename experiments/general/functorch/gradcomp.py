import torch
import torch.func as fn
import torch.autograd as ao

t = torch.randn([], requires_grad=True)
out = fn.grad(lambda x: torch.cos(x))(t)
y = torch.cos(t)
aout = ao.grad(y,t)
assert torch.allclose(aout[0], out)
