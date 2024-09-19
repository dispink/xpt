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
    val_annotations_file: str | None = None,
    val_input_dir: str | None = None,
    transform=lambda x: x,
    target_transform=lambda x: x,
    num_workers=1,
    pin_memory=True,
    test_only=False,
):
    """
    Get dataloaders (in dictionary) from split datasets.
    The dataloader for validation doesn't shuffle the data unlike the one for training.
    """

    # decide which dataset to use

    # split the dataset into train and val sets
    if test_only:
        if ispretrain:
            dataset = PretrainDataset(
                annotations_file, input_dir,
                transform=transform,
            )
        else:
            dataset = FinetuneDataset(
                annotations_file, input_dir,
                transform=transform,
                target_transform=target_transform,
            )
        dataloader = {
            "test": DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
        }
    else:
        if val_annotations_file is None:
            if ispretrain:
                dataset = PretrainDataset(
                    annotations_file, input_dir,
                    transform=transform,
                )
            else:
                dataset = FinetuneDataset(
                    annotations_file, input_dir,
                    transform=transform,
                    target_transform=target_transform,
                )
            data_train, data_val = split(dataset)
        else:
            if val_input_dir is None:
                val_input_dir = input_dir
            
            if ispretrain:
                data_train = PretrainDataset(
                    annotations_file, input_dir,
                    transform=transform,
                )
                data_val = PretrainDataset(
                    val_annotations_file, val_input_dir,
                    transform=transform,
                )
            else:
                data_train = FinetuneDataset(
                    annotations_file, input_dir,
                    transform=transform,
                    target_transform=target_transform,
                )
                data_val = FinetuneDataset(
                    val_annotations_file, val_input_dir,
                    transform=transform,
                    target_transform=target_transform,
                )

        # create dataloaders
        dataloader = {
            "train": DataLoader(
                data_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            "val": DataLoader(
                data_val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
            ),
        }

    return dataloader
