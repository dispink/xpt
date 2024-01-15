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

class PretrainDataset(Dataset):
    
    def __init__(self, annotations_file: str, input_dir: str, transform: object=None):
        """
        Initialize a Dataset object for pretraining.

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
    
class FinetuneDataset(Dataset):
    
    def __init__(self, annotations_file: str, input_dir: str, transform: object=None):
        """
        Initialize a Dataset object for finetuning.

        Args:
            annotations_file (str): Path to the annotations file.
            input_dir (str): Directory containing spectra and targets subfolders.
            transform (function, optional): Optional data transformation object for spectra. Defaults to None.

        Output:
            sample (dict): {'spe': spectrum, 'target': target}
        """
        info_path = os.path.join(input_dir, annotations_file)
        self.info_df = pd.read_csv(info_path)
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        spe_path = os.path.join(self.input_dir, 'spe', self.info_df.iloc[idx, 0])
        spe = np.loadtxt(spe_path, delimiter=',', dtype=float)
        if self.transform:
            spe = self.transform(spe)

        target_path = os.path.join(self.input_dir, 'target', self.info_df.iloc[idx, 0])
        target = np.loadtxt(target_path, delimiter=',', dtype=float)
        
        sample = {'spe': spe, 'target': target}

        return sample
    
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

def get_dataloader(ispretrain: bool, annotations_file: str, input_dir: str, batch_size: int, transform=None):
    """
    Get dataloaders (in dictionary) from split datasets.
    The dataloader for validation doesn't shuffle the data unlike the one for training.
    """
    
    # decide which dataset to use
    if ispretrain:
        dataset = PretrainDataset(annotations_file, input_dir, transform=transform)
    else:
        dataset = FinetuneDataset(annotations_file, input_dir, transform=transform)

    # split the dataset into train and val sets
    data_train, data_val = split(dataset)
    
    # create dataloaders
    dataloader = {
        'train':DataLoader(data_train, 
                           batch_size=batch_size, 
                           shuffle=True, 
                           num_workers=10,
                           pin_memory=True),
        'val':DataLoader(data_val, 
                         batch_size=batch_size, 
                         num_workers=4,
                         pin_memory=True)
        }
    
    return dataloader

if __name__ == "__main__":
    #dataset = FinetuneDataset(annotations_file='info_20240112.csv', input_dir='data/finetune/train', transform=standardize)
    #dataset = CustomImageDataset(annotations_file='data/info_20231225.csv', input_dir='data/pretrain', transform=standardize)
    #print(len(dataset))
    #print(dataset[0]['target'])
    dataloader = get_dataloader(ispretrain=False, annotations_file='info_20240112.csv', input_dir='data/finetune/train', 
                                batch_size=64, transform=standardize)
    samples = dataloader['train'].dataset[0]
    print(samples['spe'].shape)
    print(samples['target'].shape)
    #print(spe.dtype)
    #print(spe)
    #print(spe.max())