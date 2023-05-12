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
        self.df_path = df_path
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        
        self.df = pd.read_csv(self.df_path)
        label = self.df.Label.values[idx]
        filename = self.df.filename.values[idx]
        
        p_path = os.path.join(self.image_path, filename)
        
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    

if __name__ == "__main__":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([transforms.Resize((600, 600)),
                                        transforms.RandomCrop((448, 448)),
                                        transforms.Resize((448, 448)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([transforms.Resize((600, 600)),
                                        transforms.CenterCrop((448, 448)),
                                        transforms.Resize((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
    ])

    trainset = FGVSDataset("./train.csv", "./images",transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)

    testset = FGVSDataset("./test.csv", "./images",transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)