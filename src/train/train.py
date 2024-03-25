import torch
from torch import nn
from torch.nn import optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..eval.eval import evaluate


def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer: optimizer, args
):
    model = model.to(args.device)

    model.train()  # turn on train mode
    total_loss = 0.0

    # remove step_loss_list
    for samples in dataloader:
        samples = samples.to(args.device, non_blocking=True, dtype=torch.float)
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.lr_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def trainer(
    model: nn.Module,
    dataloaders: dict,
    optimizer: optimizer,
    scheduler: lr_scheduler,
    epochs: int,
    writer: SummaryWriter | None,
    args,
):
    """
    Main training loops.

    Inputs:
        model: The model to be trained.
        dataloaders: a dictionary of dataloader for training and evaluation, {'train': DataLoader, 'val': DataLoader}/
        optimizer: the optimizer for the model's parameters.
        scheduler: learning rate scheduler.
        epochs: training epochs.
    """

    for epoch in range(1, epochs + 1):
        epoch_loss = train_one_epoch(
            model=model, dataloader=dataloaders["train"], optimizer=optimizer, args=args
        )
        writer.add_scalar("train_loss", epoch_loss, epoch)

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("lr", lr, epoch)
        scheduler.step()

        if "val" in dataloaders:
            val_loss = evaluate(model=model, dataloader=dataloaders["eval"])
            writer.add_scalar("validation", val_loss, epoch)
