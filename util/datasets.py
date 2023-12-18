"""Custom dataset class for loading spe files.

It is copied from pilot/util/datasets.py, which is based on
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class CustomImageDataset(Dataset):
    
    def __init__(self, annotations_file: str, input_dir: str, transform: object=None):
        """
        Initialize a Dataset object.

        Args:
            annotations_file (str): Path to the annotations file.
            input_dir (str): Directory containing spe files.
            transform (function, optional): Optional data transformation object. Defaults to None.
        """
        self.spe_info = pd.read_csv(annotations_file)
        self.input_dir = input_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.spe_info)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.spe_info.iloc[idx, 0])
        spe = np.loadtxt(input_path, delimiter=',', dtype=int)
        if self.transform:
            spe = self.transform(spe)
        return spe
    
def standardize(spe):
    """
    spe: numpy array of shape (n_channels)
    Standardize the spectrum to have zero mean and unit variance.
    """
    mean = np.mean(spe, axis=0)
    std = np.std(spe, axis=0)
    spe = (spe - mean) / std
    return spe

def log_transform(spe):
    """
    spe: numpy array of shape (n_channels)
    Apply log transform to the spectrum.
    Add 1 to avoid log(0).
    """
    spe = np.log(spe+1)
    return spe 
    
def split(dataset: Dataset, train_ratio: float = 0.8, seed: int = 24):
    """
    Split the dataset into train and val sets.
    """
    data_train, data_val = random_split(dataset, [train_ratio, 1-train_ratio], generator=torch.manual_seed(seed))
    return data_train, data_val

def get_dataloader(annotations_file: str, input_dir: str, batch_size: int, transform=None):
    dataset = CustomImageDataset(annotations_file, input_dir, transform=transform)
    data_train, data_val = split(dataset)
    dataloader = {
        'train':DataLoader(data_train, batch_size=batch_size, shuffle=True),
        'val':DataLoader(data_val, batch_size=batch_size, shuffle=True)
        }
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataloader(annotations_file='data/info_20231214.csv', input_dir='data/spe', 
                                batch_size=64, transform=standardize)
    spe = dataloader['train'].dataset[0]
    print(spe.shape)
    print(spe.dtype)
    print(spe)
    print(spe.max())