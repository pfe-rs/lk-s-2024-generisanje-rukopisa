from calendar import c
from locale import currency
from tkinter import LabelFrame
from PIL import Image
import os
from model import *
from dataset import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen_learn_rate = 3e-4
disc_learn_rate = 3e-4
z_dim = 1
z_depth = 100
img_dim = 64
input_channels = 1
output_channels = 1
batch_size = 128
num_epochs = 500
epoch_offset = 500

train_dataset = Data('e.csv', img_dim)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
gan = GAN(input_channels, output_channels, gen_learn_rate, disc_learn_rate, device).to(device)
writer = SummaryWriter()

#gan.load_state_dict(torch.load('models/model_epoch_499.pt'))

for epoch in range(num_epochs):
    batch_idx = 0
    trainer = 0

    for image, label in train_dataloader:
        trainer += 1
        image = image.to(device)
        label = label.to(device)

        curr_batch_size = len(image)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        gan.disc_opt.zero_grad()
        gan.gen_opt.zero_grad()
        
        noise = torch.randn(curr_batch_size, input_channels * z_depth, z_dim, z_dim).to(device)
        noise = scale(noise, 0.5, 0.5)

        fake = gan.gen(noise)
        #print(image.shape)
        #print(f"fake {fake.shape}")
        
        disc_real = gan.disc(image.view(curr_batch_size, 1, img_dim, img_dim))
        disc_fake = gan.disc(fake.detach())
        
        lossD_real = gan.criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = gan.criterion(disc_fake, torch.zeros_like(disc_fake))
     
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward(retain_graph=True)
        gan.disc_opt.step()

        fake = gan.gen(noise)
        disc_fake = gan.disc(fake)
        
        lossG = gan.criterion(disc_fake, torch.ones_like(disc_fake))
        lossG.backward()
        
        gan.gen_opt.step()

        writer.add_scalar('Loss/Generator', lossG.item(), epoch)
        writer.add_scalar('Loss/Discriminator', lossD.item(), epoch)


        for i in range(curr_batch_size):
            img = fake[i]
            img = img.view(1, img_dim, img_dim)
            writer.add_image(f'slice_{batch_idx}_{i}',  img, epoch+epoch_offset)

        batch_idx += 1
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1+epoch_offset}/{num_epochs+epoch_offset}] Batch {batch_idx}/{len(train_dataset)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

    torch.save(gan.state_dict(), f'models/model_epoch_{epoch+epoch_offset}.pt')

    print(f"Epoch [{epoch+1+epoch_offset}] completed. \t Loss D: {lossD:.4f}, loss G: {lossG:.4f}")