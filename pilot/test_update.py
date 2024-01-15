"""
This is a test file to see if the model perform stablily after I update timm 
to the latest version (0.4.5 -> 0.9.12) and modify the parameter qk_scale to qk_norm 
in models_mae_loss.py.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from util.datasets import PretrainDataset, log_transform, standardize
# new loss function
from models_mae_loss import mae_vit_base_patch16
import matplotlib.pyplot as plt
import datetime
import time


def get_date():
    return datetime.date.today().strftime("%Y%m%d")

def get_dataloader(batch_size: int, transform=None):
    dataset = PretrainDataset('data/info_20231121.csv', 'data/spe', transform=transform)
    data_train, data_val = random_split(dataset, [0.8, 0.2], generator=torch.manual_seed(24))
    dataloader = {
        'train':DataLoader(data_train, batch_size=batch_size, shuffle=True),
        'val':DataLoader(data_val, batch_size=batch_size, shuffle=True)
        }
    return dataloader

def train_one_epoch(model: nn.Module, dataloader: DataLoader, lr: float, device='cuda'):
    """
    lr: learning rate
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model = model.to(device)

    model.train()  # turn on train mode
    total_loss = 0.

    # remove step_loss_list
    for samples in dataloader:
        samples = samples.to(device, non_blocking=True, dtype=torch.float)
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model: nn.Module, dataloader: DataLoader, device='cuda'):
    total_loss = 0.
    model.eval()  # turn on evaluation mode

    with torch.no_grad():
        for samples in dataloader:
            samples = samples.to(device, non_blocking=True, dtype=torch.float)
            loss, _, _ = model(samples)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def visualize(train_loss_list, val_loss_list, lr, out_dir: str):
    plt.figure()
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{out_dir}/loss_{lr}_{get_date()}.png')

def trainer(model: nn.Module, dataloader: DataLoader, lr: float, epochs: int, data: str):
    """
    Train the model for epochs. Export the training and validation loss in figure.
    Output the model's minimum validation loss.
    """
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        epoch_loss = train_one_epoch(model=model, dataloader=dataloader['train'], lr=lr)
        train_loss_list.append(epoch_loss)

        val_loss = evaluate(model, dataloader['val'])
        val_loss_list.append(val_loss)

        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:.3f} ')
        print('-' * 89)
    
    visualize(train_loss_list, val_loss_list, lr, f'results/{data}')
    return min(val_loss_list)

def main():
    # the optimal model is trained on data with standardization, lr=1e-5
    dataloader = get_dataloader(batch_size=64, transform=standardize)
    epochs = 100
    data = 'update'
    lr = 1e-5

    # reset model
    model = mae_vit_base_patch16()
    val_loss = trainer(model, dataloader, lr, epochs, data)
    print(f'val loss: {val_loss:.3f}')
    torch.save(model.state_dict(), f'models/mae_vit_base_patch16_{data}_{lr}_{get_date()}.pth')


if __name__ == '__main__':
    main()
 