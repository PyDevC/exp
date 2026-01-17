import torch
import numpy as np

# Create a Tensor
t = torch.tensor([[1, 2, 3, 4],[1, 3, 2, 3]])
arr = np.array([[1, 2, 3, 4],[1, 3, 3, 4]])
nt = torch.from_numpy(arr)
print(t)
print(nt)

# Create Tensor of shape and dtype as t
ones = torch.ones_like(t)
print(ones)

# Create a Tensor from a given shape
shape = (2, 4, )
ones = torch.ones(shape, dtype=torch.float16)
zeros = torch.zeros(shape)
rand = torch.rand(shape)

print(ones)
print(zeros)
print(rand)

# Indexing

print("first ", rand[0])
print("second", rand[1])
print("this", rand[:, 1]) # a little different from  list
print("last 4", rand[:][:-2])

# Join tensors

# With cat
tnt = torch.cat((t, nt), dim=1)
print(tnt)

# With stack
tnt = torch.stack((t, nt), dim=0)
print(tnt)

# Arithmetic operations

out = t @ t.T
print(out)
out = t.matmul(t.T)
print(out)
out = torch.matmul(t, t.T)
print(out)
out = torch.matmul(t, t.T, out=out)
print(out)

# Element wise multiplication
out = t * t
print(out)
out = t.mul(t)
print(out)

# Aggregate
out = t.sum()
print(out)
