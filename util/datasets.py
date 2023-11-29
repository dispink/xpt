import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, input_dir):
        """
        input_dir: directory with spe files
        """
        self.spe_info = pd.read_csv(annotations_file)
        self.input_dir = input_dir
        
    def __len__(self):
        return len(self.spe_info)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.spe_info.iloc[idx, 0])
        spe = np.loadtxt(input_path, delimiter=',', dtype=int)  
        return spe