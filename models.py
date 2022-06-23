from torch import nn
from torch import cat


class MelCNN(nn.Module):
    def __init__(self):
        super(MelCNN, self).__init__()

        self.conv1 = self.conv_block(1, 16)
        self.conv2 = self.conv_block(16, 32)
        self.conv3 = self.conv_block(32, 64)
        self.conv4 = self.conv_block(64, 64)

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(252 * 64, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 1),
                                    nn.Sigmoid())

    @staticmethod
    def conv_block(n_in, n_out):
        return nn.Sequential(nn.Conv2d(n_in, n_out, (3, 3)),
                             nn.ReLU(),
                             nn.AvgPool2d(2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        return self.linear(x)


class EncoderBlock(nn.Module):
    def __init__(self, f_in, f_out):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(f_in, f_out, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            # nn.Dropout2d(drop_p),
            nn.Conv2d(f_out, f_out, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            # nn.Dropout2d(drop_p)
        )

        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    def __init__(self, f_in, f_out):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(f_in, f_out, (3, 3), padding=(1, 1)),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(f_out * 2, f_out, (3, 3), padding=(1, 1)), nn.ReLU(),
            # nn.Dropout2d(drop_p),
            nn.Conv2d(f_out, f_out, (3, 3), padding=(1, 1)), nn.ReLU(),
            # nn.Dropout2d(drop_p)
        )

    def forward(self, x, e):
        x = self.upconv(x)
        return self.conv(cat((e, x), dim=1))


class Mel2MelCNN(nn.Module):
    def __init__(self):
        super(Mel2MelCNN, self).__init__()

        self.enc1 = EncoderBlock(1, 8)    # [80 1376] -> [40 688]
        self.enc2 = EncoderBlock(8, 16)   # [40 688]  -> [20 344]
        self.enc3 = EncoderBlock(16, 32)  # [20 344]  -> [10 172]
        self.enc4 = EncoderBlock(32, 64)  # [10 172]  -> [5   86]

        self.bneck = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)), nn.ReLU(),
            # nn.Dropout2d(drop_p),
            nn.Conv2d(128, 128, (3, 3), padding=(1, 1)), nn.ReLU(),
            # nn.Dropout2d(drop_p),
        )

        self.dec1 = DecoderBlock(128, 64)  # [5   86] -> [10  172]
        self.dec2 = DecoderBlock(64, 32)   # [10 172] -> [20  344]
        self.dec3 = DecoderBlock(32, 16)   # [20 344] -> [40  688]
        self.dec4 = DecoderBlock(16, 8)    # [40 688] -> [80 1376]

        self.outconv = nn.Sequential(
            nn.Conv2d(8, 1, (1, 1)), nn.Sigmoid(),
        )

    def forward(self, x):
        x_in = x

        e1, x = self.enc1(x)
        e2, x = self.enc2(x)
        e3, x = self.enc3(x)
        e4, x = self.enc4(x)

        x = self.bneck(x)

        x = self.dec1(x, e4)
        x = self.dec2(x, e3)
        x = self.dec3(x, e2)
        x = self.dec4(x, e1)

        return x_in * self.outconv(x)
