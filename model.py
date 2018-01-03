# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class GeneratorX(nn.Module):
    def __init__(self, zd=128, ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(zd, zd, 4, 1),
            nn.BatchNorm2d(zd),
            nn.LeakyReLU(0.02),

            nn.ConvTranspose2d(zd, zd//2, 5, 2),
            nn.BatchNorm2d(zd//2),
            nn.LeakyReLU(0.02),

            nn.ConvTranspose2d(zd//2, zd//4, 5, 2),
            nn.BatchNorm2d(zd//4),
            nn.LeakyReLU(0.02),

            nn.ConvTranspose2d(zd//4, ch, 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class GeneratorZ(nn.Module):
    def __init__(self, zd=128, ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, zd//4, 5, 1),
            nn.BatchNorm2d(zd//4),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd//4, zd//2, 5, 2),
            nn.BatchNorm2d(zd//2),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd//2, zd, 3, 2),
            nn.BatchNorm2d(zd),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd, zd*2, 4, 1),
            nn.BatchNorm2d(zd*2),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd*2, zd*2, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorX(nn.Module):
    def __init__(self, zd=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, zd//4, 5, 1),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd//4, zd//2, 5, 2),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd//2, zd, 3, 2),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd, zd, 4, 1),
            nn.LeakyReLU(0.02)
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorXZ(nn.Module):
    def __init__(self, zd=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(zd*2, zd*2, 1, 1),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd*2, zd, 1, 1),
            nn.LeakyReLU(0.02),

            nn.Conv2d(zd, 1, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

