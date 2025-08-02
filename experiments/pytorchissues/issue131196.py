import torch
from torch import 

torch.set_default_dtype(torch.float8_e5m2)

## https://github.com/pytorch/pytorch/issues/131196#issuecomment-2240069999
# https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815/1

## most likely somehting wrong with the the way it gets_storage_obj for a particular backend
