import torch
from torch import nn
from generator import Generator
from discriminator import Discriminator
from recognizer import Recognizer
        

class GAN(nn.Module):
    def __init__(self, in_out_channels, out_dim, glr, dlr, device):
        super().__init__()
        self.gen = Generator(in_out_channels, out_dim, device).to(device)
        self.disc = Discriminator(in_out_channels, device).to(device)
        self.rec = Recognizer()
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
            
