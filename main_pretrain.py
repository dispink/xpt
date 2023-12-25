"""Pretrain the model on the whole data for the first time.

It's copied from pilot/test_update.py. 
Basically, we use the optimal parameters from the pilot study to train the model.
The model weitghts is transfered from the optimal pilot model.
Since the training time increases ~20 times and I still has memory source, 
I decide to speed up by increasing the batch size from 64 to 256, same as BERT.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from util.datasets import CustomImageDataset, standardize, get_dataloader
from models_mae import mae_vit_base_patch16
import matplotlib.pyplot as plt
import datetime
import time


def get_date():
    return datetime.date.today().strftime("%Y%m%d")

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

def trainer(model: nn.Module, dataloader: dict, lr: float, epochs: int):
    """
    Train the model for epochs. Export the training and validation loss in figure.
    Output the model's minimum validation loss.
    dataloader: {'train': DataLoader, 'val': DataLoader}
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
    
    visualize(train_loss_list, val_loss_list, lr, f'results')
    return min(val_loss_list)

def output_df(lr_list, val_loss_list):
    df = pd.DataFrame({'lr':lr_list, 'min_val_loss':val_loss_list})
    df.to_csv(f'results/test_lr_{get_date()}.csv', index=False)

def main(lr, val_loss_best, epochs):
    """
    lr: learning rate
    val_loss_best: the best validation loss from the previous training
    """
    # the optimal model is trained on data with standardization
    dataloader = get_dataloader(annotations_file='data/info_20231225.csv', input_dir='data/pretrain', 
                                batch_size=256, transform=standardize)   

    # reset model by the optimal weights from the pilot study
    model = mae_vit_base_patch16()
    model.load_state_dict(torch.load('pilot/models/mae_vit_base_patch16_update_1e-05_20231214.pth'))

    # acquires the minimum validation loss in the training
    val_loss = trainer(model, dataloader, lr, epochs)

    # save the model if it's the best along the previous training
    if val_loss < val_loss_best:
        torch.save(model.state_dict(), f'models/mae_vit_base_patch16_{lr}_{get_date()}.pth')
    
    return val_loss


if __name__ == '__main__':
    # initialize an expected loss that triggers the save of model
    val_loss_list = [0.1]
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    
    # initialize the output txt file
    output_file = f'results/test_lr_{get_date()}.csv'
    with open(output_file, 'w') as f:
        f.write('lr,min_val_loss\n')

    for lr in lr_list:
        # train the model and get the val_loss (minimun in each training)
        val_loss = main(lr, min(val_loss_list), epochs=30)
        val_loss_list.append(val_loss)

        # output the lr and min_val_loss
        with open(output_file, 'a') as f:
            f.write(f'{lr},{val_loss}\n')

    # remove the initialized loss
    #del val_loss_list[0]

    # output the dataframe
    #output_df(lr_list, val_loss_list)
 