"""Pretrain the model on the whole data using schedule lr.

It's copied from main_pretrain_schlr.py, but with setting more similar to the base model 
finetuning for Image Classification in BEiT paper, which is followed by MAE paper.
The constant lr after warmup is removed. lr list, warmup epochs, weight decay, min. lr, optimizer
The lr scheduler contains: warmup using LinearLR and decay using CosineAnnealingLR.
The batch size is 256, same as BERT.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, optimizer
from util.datasets import FinetuneDataset, standardize, split
from models_regressor import mae_vit_base_patch16
from engine_finetune import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import datetime
import time

def get_date():
    return datetime.date.today().strftime("%Y%m%d")

def get_dataloader(annotations_file: str, input_dir: str, batch_size: int, transform=None):
    dataset = FinetuneDataset(annotations_file, input_dir, transform=transform)
    data_train, data_val = split(dataset)
    dataloader = {
        'train':DataLoader(data_train, 
                           batch_size=batch_size, 
                           shuffle=True, 
                           num_workers=10,
                           pin_memory=True),
        'val':DataLoader(data_val, 
                         batch_size=batch_size, 
                         num_workers=4,
                         pin_memory=True)
        }
    return dataloader

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

def trainer(model: torch.nn.Module, 
            dataloader: dict,
            criterion: torch.nn.Module, 
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
                                     optimizer=optimizer,
                                     criterion=criterion)
        train_loss_list.append(epoch_loss)

        val_loss = evaluate(model=model, dataloader = dataloader['val'], criterion=criterion)
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
    
    batch_size = 256
    dataloader = get_dataloader(annotations_file='info_20240112.csv', input_dir='data/finetune/train', 
                                batch_size=batch_size, transform=standardize)   

    # load modified pre-trained model
    model = mae_vit_base_patch16(pretrained=True)

    # set the optimizer and scheduler
    warmup_epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
    criterion = torch.nn.MSELoss()
    scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor=.5, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-8)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], 
                                          milestones=[warmup_epochs])
    
    # acquires the last validation loss in the training
    val_loss = trainer(model, dataloader, criterion, optimizer, scheduler, epochs, notes)

    # save the model if it's the best along the previous training
    if val_loss < val_loss_best:
        torch.save(
            {
                'warmup epoch': warmup_epochs,
                'total epoch': epochs,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
                }, 
            f'models/mae_base_patch16_{notes}_{get_date()}.pth')
    
    return val_loss


if __name__ == '__main__':
    # initialize an expected loss that triggers the save of model
    val_loss_list = [0.1]
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
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