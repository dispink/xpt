import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, dataloader: DataLoader, mask_only=False, device="cuda"):
    total_loss = 0.0
    # turn on evaluation mode
    model.eval()

    with torch.no_grad():
        for samples in dataloader:
            samples = samples.to(device, non_blocking=True, dtype=torch.float)
            loss, _, _ = model(samples, mask_only=mask_only)
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
    data_train = torch.empty((0, 2048))
    data_val = torch.empty((0, 2048))

    with torch.no_grad():
        for samples_train, samples_val in zip(dataloader["train"], dataloader["val"]):
            data_train = torch.cat((data_train, samples_train), 0)
            data_val = torch.cat((data_val, samples_val), 0)

        mean = data_train.mean()
        mse = get_mse(mean, data_val)

    return mse


def finetune_evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = "cuda",
    reverse_pred=None
):
    total_loss = 0.0
    model.eval()  # turn on evaluation mode

    for batch in dataloader:
        samples = batch["spe"].to(device, non_blocking=True, dtype=torch.float)
        targets = batch["target"].to(
            device, non_blocking=True, dtype=torch.float)

        with torch.no_grad():
            preds = model(samples)
            if reverse_pred:
                preds = reverse_pred(preds)
            loss = criterion(preds, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def finetune_evaluate_base(dataloader, mean, criterion):
    """
    Calculate the MSE of the base model, 
    i.e., the model predicts only the mean of the target values.
    """
    total_loss = 0.0

    for batch in dataloader:
        targets = batch["target"]

        with torch.no_grad():
            preds = torch.full_like(targets, mean.item())
            loss = criterion(preds, targets)

        total_loss += loss.item()

    return total_loss / len(dataloader)
