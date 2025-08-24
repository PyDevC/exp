import torch
from torch.autograd.functional import jacobian

A = torch.randn(2, 3, 3)
b = torch.randn(2, 3)

def func(A, b):
    x = torch.linalg.lstsq(A, b, )
    return x

print(jacobian(func, (A, b), vectorize=True, strategy="reverse-mode"))
# succeed
print(jacobian(func, (A, b), vectorize=True, strategy="forward-mode"))
# fail
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (6x3 and 2x3)
