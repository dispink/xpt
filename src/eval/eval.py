import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, dataloader: DataLoader, device="cuda"):
    total_loss = 0.0
    # turn on evaluation mode
    model.eval()

    with torch.no_grad():
        for samples in dataloader:
            samples = samples.to(device, non_blocking=True, dtype=torch.float)
            loss, _, _ = model(samples)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def standardize_targets(targets: torch.Tensor, preds: torch.Tensor):
    """
    Standardize the targets and preds.
    targets: (batch_size, 1, 2)
    preds: (batch_size, 1, 2)
    """

    mean = targets.mean(dim=0)
    std = targets.std(dim=0)
    targets = (targets - mean) / std
    preds = (preds - mean) / std

    return targets, preds


@torch.no_grad()
def finetune_evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader, device: torch.device = 'cuda'):
    total_loss = 0.
    model.eval()  # turn on evaluation mode

    for batch in dataloader:
        samples = batch['spe'].to(device, non_blocking=True, dtype=torch.float)
        targets = batch['target'].to(
            device, non_blocking=True, dtype=torch.float)

        with torch.cuda.amp.autocast():
            preds = model(samples)
            targets, preds = standardize_targets(targets, preds)
            loss = criterion(preds, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)
