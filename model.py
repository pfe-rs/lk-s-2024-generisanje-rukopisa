from matplotlib.pylab import f
import numpy as np
import pandas as pd
import matplotlib as plib
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, color_dim, out_dim, device):
        self.device = device
        self.kernel_size = 4
        super().__init__()
        ngf = 32
        #self.kernel = nn.Parametar(torch.randn(broj_slova, z_dim))
        self.gen = nn.Sequential(            
            nn.ConvTranspose2d(color_dim, ngf * 8, self.kernel_size, 1, 0, bias=False, device=self.device),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, self.kernel_size, 2, 1, bias=False, device=self.device),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, self.kernel_size, 2, 1, bias=False, device=self.device),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 2, ngf, self.kernel_size, 2, 1, bias=False, device=self.device),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf, out_dim, self.kernel_size, 2, 1, bias=False ,device=self.device),
            nn.Tanh()
        ).to(self.device)

    def forward(self, noise):
        #z = noise/2* self.kernel[slovo]
        return self.gen(noise)
        
class Discriminator(nn.Module):
    def __init__(self, color_dim, device):
      self.device = device
      self.kernel_size = 4
      super().__init__()
      ndf = 32
      self.disc = nn.Sequential(       
            nn.Conv2d(1, ndf, self.kernel_size, 2, 1, bias=False, device=self.device),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, self.kernel_size, 2, 1, bias=False, device=self.device),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, self.kernel_size, 2, 1, bias=False, device=self.device),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, self.kernel_size, 2, 1, bias=False, device=self.device),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, color_dim, self.kernel_size, 1, 0, bias=False, device=self.device),
            nn.Sigmoid()

      ).to(self.device)

    def forward(self, img):
        return self.disc(img)

    
class GAN(nn.Module):
    def __init__(self, img_dim, z_dim, glr, dlr, device):
        super().__init__()
        self.gen = Generator(img_dim, z_dim, device)
        self.disc = Discriminator(img_dim, device)
        self.gen_learn_rate = glr
        self.disc_learn_rate = dlr
        self.device = device

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=self.gen_learn_rate)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=self.disc_learn_rate)
        self.criterion = nn.BCELoss()