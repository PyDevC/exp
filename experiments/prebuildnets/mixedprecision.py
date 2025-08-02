import torch
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.LinearK_to_t = torch.nn.Linear(1000,10)
        self.Lineart_to_K = torch.nn.Linear(10,1000)
        self.relu = torch.nn.ReLU()

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        x = self.LinearK_to_t(x)
        x = self.relu(x)
        x = self.Lineart_to_K(x)
        x = self.relu(x)
        x = self.LinearK_to_t(x)
        x = self.relu(x)
        return x
