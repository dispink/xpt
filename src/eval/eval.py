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


def get_mse(pred_values, true_values):
    # all in tensor
    loss = (true_values - pred_values) ** 2
    loss = loss.mean()

    return loss.item()


def evaluate_base(dataloader):
    """
    Calculate the MSE of the base model, 
    i.e., the model predicts only the mean of the target values.
    """
    data = torch.empty((0, 2048))

    with torch.no_grad():
        for samples in dataloader:
            data = torch.cat((data, samples), 0)
        mean = data.mean()
        mse = get_mse(mean, data)

    return mse


def finetune_evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = "cuda",
):
    total_loss = 0.0
    model.eval()  # turn on evaluation mode

    for batch in dataloader:
        samples = batch["spe"].to(device, non_blocking=True, dtype=torch.float)
        targets = batch["target"].to(
            device, non_blocking=True, dtype=torch.float)

        with torch.no_grad():
            preds = model(samples)
            loss = criterion(preds, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)
