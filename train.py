from PIL import Image
import os
from network import *
from dataset import *

device = 'cpu'
learn_rate = 3e-4
z_dim = 32
img_dim = 28*28
batch_size = 32
num_epochs = 1

df = pd.read_csv('english.csv')
train_dataset = Data('english.csv')
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)

print(len(df))

gen = Generator(img_dim, z_dim).to(device)
disc = Discriminator(z_dim).to(device)
z = torch.randn(batch_size, z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=learn_rate)
disc_opt = torch.optim.Adam(disc.parameters(), lr=learn_rate)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for image, label in train_dataloader:
        print(image.shape, label)
       
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_opt.zero_grad()
        fake = gen(torch.randn(batch_size, z_dim).to(device))
        disc_real = disc(image).view(1, -1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(1, -1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward(retain_graph=True)
        disc_opt.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        gen_opt.zero_grad()
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        lossG.backward()
        gen_opt.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(df)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
