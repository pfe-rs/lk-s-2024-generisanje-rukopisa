import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_dim, device):
        self.device = device
        self.kernel_size = 4
        self.in_channels = in_channels
        super().__init__()
        
        ngf = 32
        self.embedding = nn.Embedding(2, in_channels * 50)
        self.gen = nn.Sequential(            
            
            nn.ConvTranspose2d(in_channels * 50, ngf * 8, self.kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, out_dim, self.kernel_size, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, label):
        
        emb_label = self.embedding(label)
        emb_label = emb_label.view(-1, self.in_channels * 50, 1, 1)
        #print(label)
        #print(emb_label)
        #print('---------------------------------')
        

        return self.gen(noise/2 + emb_label*2)
