import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("INIT: ", x.size())
        x = self.flatten(x)
        print("flatten", x.size())
        x = self.linear1(x)
        print("linear1", x.size())
        x = self.relu(x)
        print("relu", x.size())
        x = self.linear2(x)
        print("linear2", x.size())
        x = self.relu(x)
        print("relu", x.size())
        out = self.linear3(x)
        print("linear3", out.size())
        return out

model = NN()
x = torch.rand((128, 28 , 28))
out = model(x)
pred_out = nn.Softmax(dim=1)(out)
y_ = pred_out.argmax(1)
print(y_.size())
