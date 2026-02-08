import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 3, 4])
print(a.matmul(b.T))
print(b.matmul(a.T))
