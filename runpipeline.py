import data.datasets as datasets
import models.CNN as CNN
import pipeline.trainer as trainer

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

data = datasets.get_fashionmnist()
train_loader = DataLoader(data, batch_size=8, shuffle=True)

criterion = CrossEntropyLoss()
model = CNN.CNNModelScratch()
optimizer = Adam(model.parameters())

trainer.train(model, train_loader, criterion, optimizer, device="cuda")
