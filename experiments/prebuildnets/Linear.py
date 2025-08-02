import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(3,3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x

class LinearThreeLayer(nn.Module):
    def __init__(self):
        super(LinearThreeLayer, self).__init__()
        self.layer1 = nn.Linear(1000,10000)
        self.layer2 = nn.Linear(10000, 1000)
        self.layer3 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x
