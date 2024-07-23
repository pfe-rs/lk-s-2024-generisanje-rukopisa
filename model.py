from matplotlib.pylab import f
import numpy as np
import pandas as pd
import matplotlib as plib
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_dim, device):
        self.device = device
        self.kernel_size = 4
        self.in_channels = in_channels
        super().__init__()
        ngf = 32
        self.embedding = nn.Embedding(62, in_channels * 50)
        #self.kernel = nn.Parametar(torch.randn(broj_slova, z_dim))
        self.gen = nn.Sequential(            

            nn.ConvTranspose2d(in_channels * 50, ngf * 8, self.kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf * 2, ngf, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( ngf, out_dim, self.kernel_size, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, label):

        emb_label = self.embedding(torch.tensor(label))
        emb_label = emb_label.view(-1, self.in_channels * 50, 1, 1)
        
        return self.gen(emb_label + noise)
        
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

    def forward(self, img):
        return self.disc(img)

    
class GAN(nn.Module):
    def __init__(self, in_out_channels, out_dim, glr, dlr, device):
        super().__init__()
        self.gen = Generator(in_out_channels, out_dim, device).to(device)
        self.disc = Discriminator(in_out_channels, device).to(device)
        self.gen_learn_rate = glr
        self.disc_learn_rate = dlr
        self.device = device

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=self.gen_learn_rate)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=self.disc_learn_rate)
        self.criterion = nn.BCELoss()

    def scale(self, tensor, homothety_coeff, translation_coeff):
        return tensor * homothety_coeff + translation_coeff

    def compress(self, labels):
        clabels = []
        for label in labels:
            if(ord(label) >= 97 and ord(label) <= 122):
                label = ord(label) - 97
                clabels.append(label)
            elif(ord(label) >= 65 and ord(label) <= 90):
                label = ord(label) - 39
                clabels.append(label)
            else:
                label = ord(label) + 4
                clabels.append(label)

        return clabels
            
