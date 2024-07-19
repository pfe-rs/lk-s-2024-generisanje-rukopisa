from matplotlib.pylab import f
import numpy as np
import pandas as pd
import matplotlib as plib
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, img_dim, z_dim):
        super().__init__()
        self.kernel = nn.Parametar(torch.randn(broj_slova, z_dim))
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, slovo):
        z = noise/2* self.kernel[slovo]
        return self.gen(z)
    
class Discriminator(nn.Module):
    def __init__(self, img_dim):
      super().__init__()
      self.disc = nn.Sequential(
          nn.Linear(img_dim, 128),
          nn.LeakyReLU(0.1, True),
          nn.Dropout(0.3),
          nn.Linear(128, 1),
          nn.Sigmoid()
      )

    def forward(self, img):
        return self.disc(img)

    
class GAN(nn.Module):
    def __init__(self, img_dim, z_dim):
        super().__init__()
        self.gen = Generator(img_dim, z_dim)
        self.disc = Discriminator(img_dim)
        self.learn_rate = 3e-4

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learn_rate)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.learn_rate)
        self.criterion = nn.BCELoss()