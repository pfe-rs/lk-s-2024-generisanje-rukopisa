from calendar import c
from locale import currency
from PIL import Image
import os
from model import *
from dataset import *
from torch.utils.tensorboard import SummaryWriter

#print(torch.cuda.is_available())
device = torch.device('cpu')


gen_learn_rate = 3e-4
disc_learn_rate = 3e-4
z_dim = 1
z_depth = 50
img_dim = 64
input_channels = 1
output_channels = 1
batch_size = 32
num_epochs = 200
epoch_offset = 0

train_dataset = Data('english.csv', img_dim)
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

        #print(label)
        label = gan.compress(label)
        #print(label)
        label = torch.tensor(label).to(device)

        curr_batch_size = len(image)
        zeros = torch.zeros(curr_batch_size, dtype=torch.int32).to(device)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        gan.disc_opt.zero_grad()
        gan.gen_opt.zero_grad()
        noise = torch.randn(curr_batch_size, input_channels * z_depth, z_dim, z_dim)
        noise = gan.scale(noise, 1, 0.5).to(device)

        fake = gan.gen(noise, label).to(device)
       
        disc_real = gan.disc(image.view(curr_batch_size, 1, img_dim, img_dim))
        disc_fake = gan.disc(fake)
        
        lossD_real = gan.criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = gan.criterion(disc_fake, torch.zeros_like(disc_fake))
     
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward(retain_graph=True)
        gan.disc_opt.step()

        fake = gan.gen(noise, label)
        #print(fake.shape)
        disc_fake = gan.disc(fake)
        
        lossG = gan.criterion(disc_fake, torch.ones_like(disc_fake))
        lossG.backward()
        
        gan.gen_opt.step()

        writer.add_scalar('Loss/Generator', lossG.item(), epoch)
        writer.add_scalar('Loss/Discriminator', lossD.item(), epoch)

        #print(fake.shape)
        for i in range(curr_batch_size):
            img = fake[i]
            img = img.view(output_channels, img_dim, img_dim)
            writer.add_image(f'label: {label[i]}',  img, global_step=epoch+epoch_offset)

        batch_idx += 1
        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1+epoch_offset}/{num_epochs+epoch_offset}] Batch {batch_idx * curr_batch_size}/{len(train_dataset)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

    torch.save(gan.state_dict(), f'models/model_epoch_{epoch+epoch_offset}.pt')

    print(f"Epoch [{epoch+1+epoch_offset}] completed. \t\t\t\t Loss D: {lossD:.4f}, loss G: {lossG:.4f}")