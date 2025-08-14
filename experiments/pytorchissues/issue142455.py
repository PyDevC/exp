import torch

self = torch.full((5, 5, 5, 5), 3.5e+35, dtype=torch.double)
padding = [-1, -1, -1, -1 ]
torch.ops.aten.reflection_pad2d(self, padding)
