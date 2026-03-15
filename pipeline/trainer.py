import torch
import tqdm

def train(model, train_loader, criterion, optimizer, num_epoch=10, device="cpu"):
    model.train()
    model = model.to(device)

    for epoch in range(num_epoch):
        loop = tqdm.tqdm(train_loader)
        for data, label in loop:
            data = data.to(device)
            label = label.to(device)

            out = model(data)
            loss = criterion(out, label)

            optimizer.zero_grad()
            optimizer.step()

        print(f"{loss.item()=}")
