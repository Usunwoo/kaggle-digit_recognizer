import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class CustomImageDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return len(self.data)

def get_loaders(batch_size, custom_data = False):
    if custom_data:
        train_df = pd.read_csv("./data/digit-recognizer/train.csv")
        train_X, train_y = torch.FloatTensor(train_df.values[:, 1:]), torch.tensor(train_df.values[:, :1])
        train_X, train_y = train_X.reshape(-1, 28, 28), train_y.squeeze()

        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)

        train_set = CustomImageDataset(
                        data=train_X, 
                        target=train_y
                        )
        valid_set = CustomImageDataset(
                        data=valid_X, 
                        target=valid_y
                        )

    else:
        path='./data'
        train_set = datasets.MNIST(root=path,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
        valid_set = datasets.MNIST(root=path,
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, valid_loader
