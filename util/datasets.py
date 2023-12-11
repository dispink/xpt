import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, input_dir, std=True):
        """
        input_dir: directory with spe files
        """
        self.spe_info = pd.read_csv(annotations_file)
        self.input_dir = input_dir
        self.std = std
        
    def __len__(self):
        return len(self.spe_info)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.spe_info.iloc[idx, 0])
        spe = np.loadtxt(input_path, delimiter=',', dtype=int)
        if self.std:
            spe = self.standardize(spe)
        return spe
    
    def standardize(self, spe):
        """
        spe: numpy array of shape (n_channels)
        Standardize the spectrum to have zero mean and unit variance.
        """
        mean = np.mean(spe, axis=0)
        std = np.std(spe, axis=0)
        spe = (spe - mean) / std
        return spe
    
    
if __name__ == "__main__":
    dataset = CustomImageDataset(annotations_file='data/info_20231121.csv', input_dir='data/spe', std=True)
    spe = dataset[0]
    print(spe.shape)
    print(spe.dtype)
    print(spe)
    print(spe.max())