import torch
from typing import OrderedDict
import torch.nn as nn

model = nn.Sequential(
            nn.Linear(100,200),
            nn.Linear(200,10)
        )

print(model)
