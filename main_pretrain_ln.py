"""Pretrain the model on the whole data using log transform.

Basically, we use the optimal parameters from the pilot study to train the model.
The model weitghts is transfered from the optimal pilot model (log transform).
Since the training time increases ~20 times and I still has memory source, 
I decide to speed up by increasing the batch size from 64 to 256, same as BERT.

It does not converge. The loss is even increasing.

Outputs:
results/loss_0.0001_20231218.png
models/mae_vit_base_patch16_0.0001_20231218.pth
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from util.datasets import log_transform, get_dataloader
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

def trainer(model: nn.Module, dataloader: DataLoader, lr: float, epochs: int):
    """
    Train the model for epochs. Export the training and validation loss in figure.
    Output the model's minimum validation loss.
    dataloader includes train and val dataloaders
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

#def output_df(lr_list, val_loss_list):
#    df = pd.DataFrame({'lr':lr_list, 'min_val_loss':val_loss_list})
#    df.to_csv(f'results/test_lr_{get_date()}.csv', index=False)

def main():
    # the optimal model is trained on data with log transform, lr=1e-5
    dataloader = get_dataloader(annotations_file='data/info_20231214.csv', input_dir='data/spe', 
                                batch_size=256, transform=log_transform)
    epochs = 100
    val_loss_list = []
    lr = 1e-4

    # reset model
    model = mae_vit_base_patch16()
    model.load_state_dict(torch.load('pilot/models/mae_vit_base_patch16_ln_0.0001_20231212.pth'))
    val_loss = trainer(model, dataloader, lr, epochs)
    val_loss_list.append(val_loss)

    torch.save(model.state_dict(), f'models/mae_vit_base_patch16_{lr}_{get_date()}.pth')


if __name__ == '__main__':
    main()
 