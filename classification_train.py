import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import MelClassificationDataset, read
from models import MelCNN


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


lr = 1e-3
batch_size = 8
epochs = 20
name = 'melcnn.pth'


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.5).int() == y.int()).float().sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")


train_clean_list = read()
train_noisy_list = read(cls='noisy')

val_clean_list = read('val')
val_noisy_list = read('val', 'noisy')

train = {
    'path': train_clean_list + train_noisy_list,
    'label': [0] * 12000 + [1] * 12000
}
val = {
    'path': val_clean_list + val_noisy_list,
    'label': [0] * 2000 + [1] * 2000
}
train = pd.DataFrame(train)
val = pd.DataFrame(val)

train = MelClassificationDataset(train)
val = MelClassificationDataset(val)

train = DataLoader(train, batch_size=batch_size, shuffle=True)
val = DataLoader(val, batch_size=batch_size, shuffle=False)


model = MelCNN().to(device)


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=0,
                             amsgrad=False)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train, model, loss_fn, optimizer)
    val_loop(val, model, loss_fn)
print("Done!")

torch.save(model, name)
