import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x ** 2 + y ** 3

print(f"z shaped: {z.shape}")

z.backward()
dx = x.grad
dy = y.grad

print(f"grad of x: {dx}, grad of y: {dy}")
print(f"grad of x: {x.device}, grad of y: {y.device}")
