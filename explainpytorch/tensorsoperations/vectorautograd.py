import torch

x = torch.tensor([1, 2, 3, 4], requires_grad=True, dtype=torch.float32)
y = torch.tensor([6, 4, 5, 6], requires_grad=True, dtype=torch.float32)

print(x.dtype)
z0 = (x**2 + y**3)
z = z0.sum()

print(z)
print(z0)

z.backward()
dy_dx = x.grad
dz_dy = y.grad

print(dy_dx, dz_dy)
