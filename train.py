from calendar import c
from locale import currency
from PIL import Image
import os
from network import *
from dataset import *

device = 'cpu'
learn_rate = 3e-4
z_dim = 32
img_dim = 28
batch_size = 32
num_epochs = 1

train_dataset = Data('english.csv', img_dim)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
gan = GAN(img_dim*img_dim, z_dim).to(device)


for epoch in range(num_epochs):
    batch_idx = 0

    for image, label in train_dataloader:
        curr_batch_size = len(image)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        gan.disc_opt.zero_grad()
        
        fake = gan.gen(torch.randn(batch_size, z_dim).to(device))
        print(image.shape)
        disc_real = gan.disc(image.view(curr_batch_size, -1))
        disc_fake = gan.disc(fake.view(curr_batch_size, -1).detach())
        
        lossD_real = gan.criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = gan.criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward(retain_graph=True)
        
        gan.disc_opt.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        gan.gen_opt.zero_grad()
        output = gan.disc(fake.view(curr_batch_size, -1))
        
        lossG = gan.criterion(output, torch.ones_like(output))
        lossG.backward()
        
        gan.gen_opt.step()


        batch_idx += 1
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_dataset)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

    torch.save(gan.state_dict(), f'model_epoch_{epoch}.pt')