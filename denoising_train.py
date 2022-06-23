import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import MelDenoisingDataset, read, dice_loss
from models import Mel2MelCNN, EncoderBlock, DecoderBlock


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


lr = 1e-3
batch_size = 8
epochs = 32
name = 'mel2melcnn'


def he_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    history = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        history.append(loss.item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return history


def val_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    val_loss, mse = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            mse += nn.functional.mse_loss(pred, y).sum().item()

    val_loss /= num_batches
    mse /= num_batches
    print(f"Test Error: \n MSE: {mse:>0.4f}, Avg loss: {val_loss:>8f} \n")
    return [val_loss]


train_clean_list = read()
train_noisy_list = read(cls='noisy')

val_clean_list = read('val')
val_noisy_list = read('val', 'noisy')

train = {
    'noisy': train_noisy_list,
    'clean': train_clean_list,
}
val = {
    'noisy': val_noisy_list,
    'clean': val_clean_list,
}

train = pd.DataFrame(train)
val = pd.DataFrame(val)

train = MelDenoisingDataset(train)
val = MelDenoisingDataset(val)

train = DataLoader(train, batch_size=batch_size, shuffle=True)
val = DataLoader(val, batch_size=batch_size, shuffle=False)


def plot(train, val, name):
    x_1 = np.array(list(range(len(train))))
    x_2 = np.array(list(range(len(val)))) * (12000 // batch_size + 1)
    plt.plot(x_1, train, 'g--', label="train")
    plt.plot(x_2, val, 'r-o', label="val")
    plt.legend()
    plt.ylim([.0, .1])
    plt.savefig(name, bbox_inches='tight')
    plt.clf()


def vis(model, name, ep):
    with torch.no_grad():
        x, y = val.dataset[0]
        x = x.to(device)
        y = y.to(device)
        x = x.view((1, 1, 80, 1376))
        y = y.view((1, 1, 80, 1376))
        y_hat = model(x)

        img = np.zeros((80 * 3, 1376))
        img[:80, :] = x.view(80, 1376).cpu().numpy()
        img[80:80 * 2, :] = y_hat.view(80, 1376).cpu().numpy()
        img[80 * 2:, :] = y.view(80, 1376).cpu().numpy()

        plt.imsave(f'model/{name}/log/{ep}.jpg', img, cmap='jet')


model = Mel2MelCNN().to(device)
model.apply(he_init)


loss_fn = dice_loss
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=0.1,
                             amsgrad=True)

scheduler_plato = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             factor=0.1,
                                                             patience=0)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=2)

train_history = []
val_history = []

if not os.path.isdir(f'model/{name}/log/'):
    os.makedirs(f'model/{name}/log/')

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_h = train_loop(train, model, loss_fn, optimizer)
    val_h = val_loop(val, model, loss_fn)
    train_history += train_h
    val_history += val_h
    plot(train_history, val_history, f'model/{name}/log/history.jpg')
    scheduler_plato.step(val_h[0])
    scheduler_cosine.step()
    vis(model, name, t)
    torch.save(model, f'model/{name}/{name}_e{t}.pth')
print("Done!")

torch.save(model, f'model/{name}/{name}.pth')
