from gan import *
from dataset import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from fid import FID 

print(torch.cuda.is_available())
print(torch.cuda.current_device())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen_learn_rate = 2e-3
disc_learn_rate = 3e-4
z_dim = 1
z_depth = 50
img_dim = 64
input_channels = 1
output_channels = 1
batch_size = 32
num_epochs = 1000
epoch_offset = 0

train_dataset = Data('english.csv', img_dim)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
gan = GAN(input_channels, output_channels, gen_learn_rate, disc_learn_rate, device).to(device)
fid = FID()
writer = SummaryWriter()

#gan.load_state_dict(torch.load('models/model_epoch_199.pt'))

def train_model():
    images = []
    labels = []
    trainers = []
    prev = gan.gen.embedding.weight.data
    
    for trainer, (image, label) in enumerate(tqdm(train_dataloader)):
        trainers.append(trainer)
        images.append(image)
        labels.append(label)

    for epoch in range(num_epochs):
        batch_idx = 0

        for i in range(len(trainers)):
            trainer = trainers[i]
            image = images[i]
            label = labels[i]

            image = image.to(device)

            #print(label)
            raw_label = label
            label = gan.compress(label)
            #print(label)
            label = torch.tensor(label).to(device)

            curr_batch_size = len(image)
  
            gan.disc_opt.zero_grad()
            noise = torch.randn(curr_batch_size, input_channels * z_depth, z_dim, z_dim)
            noise = gan.scale(noise, homothety_coeff=0.25, translation_coeff=0).to(device)

            fake = gan.gen(noise, label).to(device)

            fid_value = fid.calculate_fretchet(image, fake)
            writer.add_scalar('FID', fid_value, epoch)

            disc_real = gan.disc(image.view(curr_batch_size, 1, img_dim, img_dim), label)
            disc_fake = gan.disc(fake, label)
            
            lossD_real = gan.criterion(disc_real, torch.ones_like(disc_real))
            lossD_fake = gan.criterion(disc_fake, torch.zeros_like(disc_fake))
        
            lossD = (lossD_real + lossD_fake) / 2
            lossD.backward(retain_graph=True)
            gan.disc_opt.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            gan.gen_opt.zero_grad()
            #fake = gan.gen(noise, label) ????????????? mozda treba
            disc_fake = gan.disc(fake, label)
            #fake = (fake + 1) / 2
            #rec_label = gan.rec.recognize(fake)
            
            lossG = gan.criterion(disc_fake, torch.ones_like(disc_fake))
            lossG.backward()
            
            gan.gen_opt.step()


            writer.add_scalar('Loss/Generator', lossG.item(), epoch)
            writer.add_scalar('Loss/Discriminator', lossD.item(), epoch)

            #print(fake.shape)
            for i in range(curr_batch_size):
                img = fake[i]
                img = img.view(output_channels, img_dim, img_dim)
                writer.add_image(f'label: {raw_label[i]}/epoch: {epoch}',  img, global_step=epoch+epoch_offset)

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1+epoch_offset}/{num_epochs+epoch_offset}] Batch {batch_idx * curr_batch_size}/{len(train_dataset)} \
                    Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )
        torch.save(gan.state_dict(), f'models/model_epoch_{epoch+epoch_offset}.pt')

        # Inside your training loop

        batch_idx += 1
        if (epoch+1) % 5 == 0:
            with open('debug.txt', 'a') as f:
                f.write(f"Embedding weights at five epochs before {epoch+1}: {prev}\n")
                f.write(f"Embedding weights at epoch {epoch+1}: {gan.gen.embedding.weight.data}\n")
                f.write(f"Diff of embedding weights at epoch {epoch+1}: {gan.gen.embedding.weight.data-prev}\n")
            prev = gan.gen.embedding.weight.data

        print(f"Epoch [{epoch+1+epoch_offset}] completed. \t\t\t\t Loss D: {lossD:.4f}, loss G: {lossG:.4f}")


if __name__ == '__main__':
    
    train_model()
    writer.close()