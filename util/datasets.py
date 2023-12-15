"""Custom dataset class for loading spe files.

It is copied from pilot/util/datasets.py, which is based on
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
    
if __name__ == "__main__":
    for transfom in [None, standardize, log_transform]:
        dataset = CustomImageDataset(annotations_file='data/info_20231214.csv', input_dir='data/spe', transform=transfom)
        spe = dataset[60]
        print(spe.shape)
        print(spe.dtype)
        print(spe)
        print(spe.max())