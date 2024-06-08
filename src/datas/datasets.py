import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):

    def __init__(
        self,
        annotations_file: str,
        input_dir: str,
        transform: object = lambda x: x,
    ):
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
        spe = torch.from_numpy(np.loadtxt(input_path, delimiter=",", dtype=float))
        spe = self.transform(spe)
        return spe


class FinetuneDataset(Dataset):

    def __init__(
        self,
        annotations_file: str,
        input_dir: str,
        transform: object = lambda x: x,
    ):
        """
        Initialize a Dataset object for finetuning.

        Args:
            annotations_file (str): Path to the annotations file.
            input_dir (str): Directory containing spectra and targets subfolders.
            transform (function, optional): Optional data transformation object for spectra. Defaults to None.

        Output:
            sample (dict): {'spe': spectrum, 'target': target}
        """
        self.info_df = pd.read_csv(annotations_file)
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        spe_path = os.path.join(self.input_dir, "spe", self.info_df.iloc[idx, 0])
        spe = torch.from_numpy(np.loadtxt(spe_path, delimiter=",", dtype=float))
        spe = self.transform(spe)

        target_path = os.path.join(self.input_dir, "target", self.info_df.iloc[idx, 0])
        target = torch.from_numpy(
            np.loadtxt(target_path, delimiter=",", dtype=float)
        ).unsqueeze(-1)

        sample = {"spe": spe, "target": target}

        return sample
