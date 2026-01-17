import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor


# Training data

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 8 # can be changed

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("shape of X: ", X.shape)
    print("shape of y: ", y.shape)
    break

device = 'cuda'
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
print(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer, epochs=20):
    size = len(dataloader)
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:.4f}, [{current:5d}/{size:>5d}]")
        print(f"Epoch: [{epoch}/{epochs}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            test_loss += loss_fn(out, y).item()
            print("correct: ", (out.argmax(1)==y))
            correct += (out.argmax(1) == y).type(torch.float).sum().item()

    print(f"Accuracy: {correct/size:5.f}, loss: {test_loss}")
            

train(train_dataloader, model, loss_fn, optimizer)
test(train_dataloader, model, loss_fn)
