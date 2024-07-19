from calendar import c
from locale import currency
from PIL import Image
import os
from network import *
from dataset import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


device = 'cpu'
learn_rate = 3e-3
z_dim = 32
img_dim = 28
batch_size = 32
num_epochs = 500

train_dataset = Data('e.csv', img_dim)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
gan = GAN(img_dim*img_dim, z_dim).to(device)
writer = SummaryWriter()

for epoch in range(num_epochs):
    batch_idx = 0

    for image, label in train_dataloader:
        

        curr_batch_size = len(image)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        gan.disc_opt.zero_grad()
        gan.gen_opt.zero_grad()
        
        noise = torch.randn(curr_batch_size, z_dim).to(device)

        fake = gan.gen(noise)
        #print(image.shape)
        disc_real = gan.disc(image.view(curr_batch_size, -1))
        disc_fake = gan.disc(fake.view(curr_batch_size, -1).detach())
        
        lossD_real = gan.criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = gan.criterion(disc_fake, torch.zeros_like(disc_fake))
        

        #lossD = -torch.mean(torch.log(disc_real) + torch.log(1 - disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward(retain_graph=True)
        

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        gan.disc_opt.step()

        fake = gan.gen(noise)
        disc_fake = gan.disc(fake.view(curr_batch_size, -1))
        
        lossG = gan.criterion(disc_fake, torch.ones_like(disc_fake))
        #lossG = -torch.mean(torch.log(1-disc_fake))
        lossG.backward()
        
        gan.gen_opt.step()

        for i in range(fake.size(0)):
            img = fake[i]
            img = img.view(1, 28, 28)
            writer.add_image(f'slice_{batch_idx}_{i}',  img, epoch)

        batch_idx += 1
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_dataset)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

    torch.save(gan.state_dict(), f'models/model_epoch_{epoch}.pt')

    print(f"Epoch [{epoch+1}] completed. \t Loss D: {lossD:.4f}, loss G: {lossG:.4f}")