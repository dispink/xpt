import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, dataloader: DataLoader, device="cuda"):
    total_loss = 0.0
    model.eval()  # turn on evaluation mode

    with torch.no_grad():
        for samples in dataloader:
            samples = samples.to(device, non_blocking=True, dtype=torch.float)
            loss, _, _ = model(samples)
            total_loss += loss.item()
    return total_loss / len(dataloader)
