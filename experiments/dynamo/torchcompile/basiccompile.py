import torch

def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    a = a + b
    c = torch.relu(b)
    return c

new_fn = torch.compile(fn, backend="inductor")
t = torch.randn(10000, dtype=torch.float32, device="cpu")
t1 = torch.randn(10000, dtype=torch.float32, device="cpu")
a = new_fn(t, t1)
