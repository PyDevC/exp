import torch
import torch.nn as nn

# Perform Flatten
x = torch.randn((2, 2, 2, 2, 2, 2))
t = nn.Flatten(start_dim=0, end_dim=-1)(x)
#print(x)
#print(t)

# Perform Linear Transformation
t = nn.Linear(in_features=2, out_features=32)(x)
print(t)
