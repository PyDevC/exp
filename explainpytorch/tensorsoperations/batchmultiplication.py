import torch

a = torch.randn(size=(10, 2))
b = torch.randn(size=(2, 10))

c = a @ b
print(c.shape)

print(c.reshape(-1).shape)
