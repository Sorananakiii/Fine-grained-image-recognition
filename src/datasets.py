import cv2, os
import pandas as pd
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, Subset

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pd.set_option('display.max_colwidth', 200)


class FGVSDataset(Dataset):
    
    def __init__(self, df_path, image_path, transform=None):
        self.df = pd.read_csv(df_path)
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):

        label = self.df.Label.values[idx]
        filename = self.df.filename.values[idx]
        
        p_path = os.path.join(self.image_path, filename)
        
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label