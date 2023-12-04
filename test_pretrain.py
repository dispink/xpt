"""
It's a test to train and evaluate the model. 
The workflow followes the idea of the main_pretrain.py and 
engine_pretrain.py from MAE repo (https://github.com/facebookresearch/mae).
I did modify those two files to fit our case, but the loss of the 
first btach is nan, which was not the case in our simple forward test
using one batch (test_forward.py). Therefore, I decide to build the 
training script from the scratch without the fancy and complicated 
codes like the ones in MAE repo. 

No warming ip, no gradient accumulation, no mixed precision,
no lr scheduler, no logging, no checkpointing, no parallelization
"""

import numpy as np
import torch
from torch import nn
from torch import Generator
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from util.datasets import CustomImageDataset
from models_mae import mae_vit_base_patch16
import matplotlib.pyplot as plt
import datetime
import time

def get_date():
    return datetime.date.today().strftime("%Y%m%d")

def train(model: nn.Module, dataloader: DataLoader, log_interval: int, device='cuda'):
    """
    log_interval: how many batches as a step
    """
    lr = 1e-5  # fixed learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model = model.to(device)

    model.train()  # turn on train mode
    step_loss = 0.
    step_loss_list = []
    total_loss = 0.
    #start_time = time.time()

    for batch, samples in enumerate(dataloader):
        samples = samples.to(device, non_blocking=True, dtype=torch.float)
        loss, _, _ = model(samples)
        #print(math.isfinite(loss_value))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step_loss += loss.item()
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            #lr = scheduler.get_last_lr()[0]
            #ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            #cur_loss = step_loss / log_interval
            #print(f'| epoch {epoch:3d} | {batch:5d}/{len(dataloader):5d} batches | '
            #      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            #      f'loss {cur_loss:5.2f}')
            step_loss_list.append(step_loss / log_interval)
            step_loss = 0.
            #start_time = time.time()
    return step_loss_list, total_loss / len(dataloader)

def evaluate(model: nn.Module, dataloader: DataLoader, device='cuda'):
    total_loss = 0.
    model.eval()  # turn on evaluation mode

    with torch.no_grad():
        for samples in dataloader:
            samples = samples.to(device, non_blocking=True, dtype=torch.float)
            loss, _, _ = model(samples)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def visualize(train_loss_list, val_loss_list, step_loss_list, log_interval: int, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_loss_list, label='train loss')
    axes[0].plot(val_loss_list, label='val loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    axes[1].plot(
        np.array([_ for _ in range(1, len(step_loss_list)+1)])*log_interval,
        step_loss_list,
        label='step loss'
        )
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('loss')
    axes[1].legend()
    fig.savefig(f'{out_dir}/loss_{get_date()}.png')
    

if __name__ == '__main__':

    dataset = CustomImageDataset('data/info_20231121.csv', 'data/spe')
    data_train, data_val = random_split(dataset, [0.8, 0.2], generator=Generator().manual_seed(24))
    batch_size = 64
    dataloader = {
        'train':DataLoader(data_train, batch_size=batch_size, shuffle=True),
        'val':DataLoader(data_val, batch_size=batch_size, shuffle=True)
        }

    epochs = 100
    log_interval = 10
    model = mae_vit_base_patch16()
    train_loss_list = []
    step_loss_list = []
    val_loss_list = []

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        step_losses, epoch_loss = train(model, dataloader['train'], log_interval)
        step_loss_list.extend(step_losses)
        train_loss_list.append(epoch_loss)

        val_loss = evaluate(model, dataloader['val'])
        val_loss_list.append(val_loss)

        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} ')
        print('-' * 89)
        
    torch.save(model.state_dict(), f'models/mae_vit_base_patch16_{get_date()}.pth')
    visualize(train_loss_list, val_loss_list, step_loss_list, log_interval, 'results')
 