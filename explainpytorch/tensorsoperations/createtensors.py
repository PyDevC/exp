import torch
import numpy as np

# From array
t = torch.tensor([1, 2, 3])
print(t)

# From numpy array
array = np.array([1, 2, 3])
t = torch.from_numpy(array)
print(t)
