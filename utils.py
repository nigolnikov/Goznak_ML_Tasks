import numpy as np

from glob import glob
import re

import torch
from torch.utils.data import Dataset


train_min = -1.762695313
train_max = 1.78320313


class MelClassificationDataset(Dataset):
    def __init__(self, data):
        self.df = data.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx, 0]
        label = np.zeros(1, dtype=np.float32)
        label[0] = self.df.iloc[idx, 1]

        image = np.zeros((1, 80, 1374), dtype=np.float32)
        x = np.load(path).T
        image[:, :, :x.shape[1]] = x
        image -= train_min
        image /= train_max
        return torch.from_numpy(image), torch.from_numpy(label)


class MelDenoisingDataset(Dataset):
    def __init__(self, data):
        self.df = data.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        noisy_path = self.df.iloc[idx, 0]
        clean_path = self.df.iloc[idx, 1]

        noisy_image = np.zeros((1, 80, 1376), dtype=np.float32)
        clean_image = np.zeros((1, 80, 1376), dtype=np.float32)

        x = np.load(noisy_path).T
        y = np.load(clean_path).T

        noisy_image[:, :, :x.shape[1]] = x
        noisy_image -= train_min
        noisy_image /= train_max

        clean_image[:, :, :x.shape[1]] = y
        clean_image -= train_min
        clean_image /= train_max

        noisy_image[noisy_image < 0] = 0
        noisy_image[noisy_image > 1] = 1
        clean_image[clean_image < 0] = 0
        clean_image[clean_image > 1] = 1
        return torch.from_numpy(noisy_image), torch.from_numpy(clean_image)


def read(sub='train', cls='clean'):
    return [re.sub(r'\\', r'/', x) for x in glob(f'data/{sub}/{cls}/*/*.npy')]


def dice_loss(inputs, targets, smooth=1):
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice
