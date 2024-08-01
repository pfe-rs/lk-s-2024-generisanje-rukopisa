import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, out_channels, device):
        self.device = device
        self.kernel_size = 4
        super().__init__()

        ndf = 32
        self.disc = nn.Sequential(

            nn.Conv2d(1, ndf, self.kernel_size, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, out_channels, self.kernel_size, 1, 0, bias=False),
            nn.Sigmoid()

      )

    def forward(self, x, label):

        return self.disc(x)
