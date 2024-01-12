"""Finetune the model on the downstream task.

It's partly copied from main_pretrain_schlr.py. 
Basically, I adopt the settings from MAE paper. The gradient clipping is removed.
samples: (batch_size, spe_length, 1)
preds and targets: (batch_size, 1, 2)
"""

import torch

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device = 'cuda', mask_ratio: float = 0.75):

    model = model.to(device)

    model.train()  # turn on train mode
    total_loss = 0.

    # remove step_loss_list
    for batch in dataloader:
        samples = batch['spe'].to(device, non_blocking=True, dtype=torch.float)
        targets = batch['target'].to(device, non_blocking=True, dtype=torch.float)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            preds, _ = model(samples, mask_ratio=mask_ratio)
            loss = criterion(preds, targets)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, device: torch.device = 'cuda',
             mask_ratio: float = 0.75):
    total_loss = 0.
    model.eval()  # turn on evaluation mode

    for batch in dataloader:
        samples = batch['spe'].to(device, non_blocking=True, dtype=torch.float)
        targets = batch['target'].to(device, non_blocking=True, dtype=torch.float)
        
        with torch.cuda.amp.autocast():
            preds, _ = model(samples, mask_ratio=mask_ratio)
            loss = criterion(preds, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)