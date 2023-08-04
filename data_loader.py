import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

path = "./data/"


class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y
    

def load_data(flatten=False):
    train_df = pd.read_csv("./data/train.csv")
    # train_X = torch.tensor(train_df.values[:, 1:]).reshape(-1, 28, 28)
    train_X = torch.tensor(train_df.values[:, 1:]) / 255
    train_y = torch.tensor(train_df.values[:, :1])

    if not flatten:
        train_X = train_X.reshape(-1, 28, 28)

    return train_X, train_y


def get_loaders(config):
    train_X, train_y = load_data()

    train_size = int(train_X.size(0) * config.train_ratio)
    valid_size = train_X.size(0) - train_size

    indices = torch.randperm(train_X.size(0))
    train_X, valid_X = torch.index_select(
        train_X, 
        dim=0, 
        index=indices
    ).split([train_size, valid_size], dim=0)
    train_y, valid_y = torch.index_select(
        train_y, 
        dim=0, 
        index=indices
    ).split([train_size, valid_size], dim=0)

    train_loader = DataLoader(
        dataset=MyDataset(train_X, train_y), 
        batch_size=config.batch_size, 
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=MyDataset(valid_X, valid_y), 
        batch_size=config.batch_size, 
        shuffle=False
    )

    return train_loader, valid_loader
