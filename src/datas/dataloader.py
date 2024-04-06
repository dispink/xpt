import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from .datasets import PretrainDataset, FinetuneDataset


def split(dataset: Dataset, train_ratio: float = 0.8, seed: int = 24):
    """
    Split the dataset into train and val sets.
    """
    data_train, data_val = random_split(
        dataset, [train_ratio, 1 - train_ratio], generator=torch.manual_seed(seed)
    )
    return data_train, data_val


def get_dataloader(
    ispretrain: bool,
    annotations_file: str,
    input_dir: str,
    batch_size: int,
    transform=None,
    data_transformer=None,
    args=None,
):
    """
    Get dataloaders (in dictionary) from split datasets.
    The dataloader for validation doesn't shuffle the data unlike the one for training.
    """

    # decide which dataset to use
    if ispretrain:
        dataset = PretrainDataset(
            annotations_file, input_dir, transform=transform, data_transformer=data_transformer)
    else:
        dataset = FinetuneDataset(
            annotations_file, input_dir, transform=transform, data_transformer=data_transformer)

    # split the dataset into train and val sets
    data_train, data_val = split(dataset)

    # create dataloaders
    dataloader = {
        "train": DataLoader(
            data_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            data_val, batch_size=batch_size, num_workers=4, pin_memory=True
        ),
    }

    return dataloader
