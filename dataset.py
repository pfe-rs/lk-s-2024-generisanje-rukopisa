from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms

df = pd.read_csv('english.csv')
df.head()

class Data(Dataset):
    def __init__(self, path_to_file):
        self.data = pd.read_csv(path_to_file)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.images = self.data.iloc[:, 0]
        self.captions = self.data.iloc[:, 1]

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.captions[idx]

        image = Image.open(image_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)

        # pretvoriti u torch tensor
        return image, label