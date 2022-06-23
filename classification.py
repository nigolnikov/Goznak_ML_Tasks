import numpy as np
import torch
import sys
from models import MelCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_min = -1.762695313
train_max = 1.78320313
model = torch.load('classification_model.pth')

image = np.zeros((1, 1, 80, 1374), dtype=np.float32)
x = np.load(sys.argv[1]).T
l = min(x.shape[1], 1374)
image[:, :, :, :l] = x[:, :l]
image -= train_min
image /= train_max
image = torch.from_numpy(image).to(device)

print('noisy' if (model(image) >= 0.6) else 'clean')
