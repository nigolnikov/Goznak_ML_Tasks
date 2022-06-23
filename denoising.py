import numpy as np
import torch
import sys
from models import Mel2MelCNN, EncoderBlock, DecoderBlock


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_min = -1.762695313
train_max = 1.78320313
model = torch.load('denoising_model.pth')


image = np.zeros((1, 1, 80, 1376), dtype=np.float32)
x = np.load(sys.argv[1]).T
l = min(x.shape[1], 1376)
image[:, :, :, :l] = x[:, :l]
image -= train_min
image /= train_max
image[image < 0] = 0
image[image > 1] = 1
image = torch.from_numpy(image).to(device)

clean = model(image).view(80, 1376).T.cpu().detach().numpy()
np.save(sys.argv[2], clean)
