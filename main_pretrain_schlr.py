"""Pretrain the model on the whole data using schedule lr.

It's copied from main_pretrain.py. 
Basically, I adopt the optimal parameters from the pilot study to train the model.
The lr scheduler contains: warmup using LinearLR, stable using ConstantLR, and 
decay using CosineAnnealingLR.
The batch size is increased from 64 to 256, same as BERT, to speed up the training.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, optimizer
from util.datasets import standardize, get_dataloader
from models_mae import mae_vit_base_patch16
import matplotlib.pyplot as plt
import datetime
import time


def get_date():
    return datetime.date.today().strftime("%Y%m%d")

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optimizer, device='cuda'):
    """
    lr: learning rate
    """
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

def visualize(train_loss_list, val_loss_list, out_dir: str, notes: str):
    plt.figure()
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{out_dir}/loss_{notes}_{get_date()}.png')

def output_logs(logs_list, out_dir: str, notes: str):
    with open(f'{out_dir}/logs_{notes}_{get_date()}.txt', 'w') as f:
        for logs in logs_list:
            f.write(logs + '\n')

def trainer(model: nn.Module, 
            dataloader: dict, 
            optimizer: optimizer, 
            scheduler: lr_scheduler, 
            epochs: int,
            notes: str):
    """
    Train the model for epochs. Export the training and validation loss in figure.
    Output the model's last validation loss and the logs.
    dataloader: a dictionary {'train': DataLoader, 'val': DataLoader}
    """
    train_loss_list = []
    val_loss_list = []
    logs_list = []

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        epoch_loss = train_one_epoch(model=model, 
                                     dataloader=dataloader['train'], 
                                     optimizer=optimizer)
        train_loss_list.append(epoch_loss)

        val_loss = evaluate(model, dataloader['val'])
        val_loss_list.append(val_loss)
        
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - epoch_start_time
        
        logs = f'| epoch {epoch:3d} | time: {elapsed:5.2f}s | train loss {epoch_loss:.3f} | valid loss {val_loss:.3f} | lr: {lr:.3e} '
        logs_list.append(logs)
        print('-' * 89)
        print(logs)    
        print('-' * 89)
    
    visualize(train_loss_list, val_loss_list, 'results', notes)
    output_logs(logs_list, 'results', notes)
    return val_loss_list[-1]

def main(lr, val_loss_best, epochs, notes):
    """
    lr: learning rate
    val_loss_best: the best validation loss from the previous training
    """

    # the optimal model is trained on data with standardization
    dataloader = get_dataloader(annotations_file='data/info_20231225.csv', input_dir='data/pretrain', 
                                batch_size=256, transform=standardize)   

    # reset model
    model = mae_vit_base_patch16()

    # set the optimizer and scheduler
    warmup_epochs = 10
    stable_epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=.5, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=stable_epochs)
    scheduler3 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs - stable_epochs)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], 
                                          milestones=[warmup_epochs, warmup_epochs + stable_epochs])
    #scheduler = lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
    
    # acquires the last validation loss in the training
    val_loss = trainer(model, dataloader, optimizer, scheduler, epochs, notes)

    # save the model if it's the best along the previous training
    if val_loss < val_loss_best:
        torch.save(model.state_dict(), f'models/mae_vit_base_patch16_{notes}_{get_date()}.pth')
    
    return val_loss


if __name__ == '__main__':
    # initialize an expected loss that triggers the save of model
    val_loss_list = [0.1]
    lr_list = [1e-5, 1e-6]
    # notes for naming output files
    notes = 'lr'

    # initialize the output txt file
    output_file = f'results/tuning_{notes}_{get_date()}.csv'
    with open(output_file, 'w') as f:
        f.write('lr,min_val_loss\n')

    for lr in lr_list:
        # further naming details
        notes = notes.split('_')[0] + f'_{lr}'

        # train the model and get the val_loss (minimun in each training)
        val_loss = main(lr, min(val_loss_list), epochs=100, notes=notes)
        val_loss_list.append(val_loss)

        # output the lr and min_val_loss
        with open(output_file, 'a') as f:
            f.write(f'{lr},{val_loss}\n')
