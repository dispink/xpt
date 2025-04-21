import torch
from torch import nn
from torch.utils.data import DataLoader


def inference(model: nn.Module, dataloader: DataLoader, device="cuda"):
    model.to(device)

    # turn on evaluation mode
    model.eval()
    # create a tensor to store the predictions
    predictions = torch.empty((0, 1)).to(device)

    with torch.no_grad():
        for samples in dataloader:
            samples = samples.to(device, non_blocking=True, dtype=torch.float)
            outputs = model(samples)

            # append outputs to the predictions tensor
            predictions = torch.cat((predictions, outputs), 0)

    # Return the raw predictions list
    return predictions


class DeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        output = self.std * x + self.mean

        # Clamps numbers into the range [0, 100] as the wt% values
        output = torch.clamp(output, min=0, max=100)
        return output
