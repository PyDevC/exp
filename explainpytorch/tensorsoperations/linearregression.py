import pandas as pd
from sklearn.preprocessing import Normalizer
import torch
from torch.optim import SGD

df = pd.DataFrame({
    "area": [100, 200, 33,333,3333],
    "age": [1,1,1,1,8],
    "price": [12000,120000,544511,545454,4554],
})

print(df.head())

## Normalize dataset
norm = Normalizer()
norm = norm.fit(df)
out = norm.transform(df, copy=True)
df = pd.DataFrame(out, columns=df.columns)

print(df.head())
print(df[['area', 'age']])

## Convert the df to tensor
x = torch.tensor(df[["area", "age"]].values, dtype=torch.float32)
y = torch.tensor(df["price"].values, dtype=torch.float32)
print(f"X: {x}\nY: {y}")


w = torch.randn(size=(2, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(w, b)

## Calculate prediction
def prediction(w, b):
    pred = x @ w + b
    error = (y - pred) ** 2
    loss = error.mean()
    print(f"Loss = {loss.item()}")
    return loss, w, b

## Backward
def update(loss, w, b):
    loss.backward()
    dw = w.grad
    db = b.grad
    return w, b, loss, dw, db

lr = 0.2

## Adjust the weights and bias
def adjust(w, b, dw, db):
    with torch.no_grad():
        w = w - lr * dw
        b = b - lr * db
    return w, b

for i in range(100):
    loss, w, b = prediction(w, b)
    w, b, loss, dw, db = update(loss, w, b)
    w, b = adjust(w, b, dw, db)

