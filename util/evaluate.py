import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate(model: nn.Module, dataloader: DataLoader, device='cuda'):
    total_loss = 0.
    model.eval()  # turn on evaluation mode

    with torch.no_grad():
        for samples in dataloader:
            samples = samples.to(device, non_blocking=True, dtype=torch.float)
            loss, _, _ = model(samples)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_spe(model, spe_arr):
    """
    Use the mae model to evaluate the spectra directly.
    Input: mae-like model, spetra in numpy array
    Output: loss, pred, mask
    """
    model.eval()
    with torch.no_grad():
        torch.manual_seed(24)
        spe = torch.from_numpy(spe_arr).to('cuda')
        return model(spe.unsqueeze(0).float())

def get_mse(true_values, pred_values):
    # all in tensor
    loss = (true_values - pred_values) ** 2
    loss = loss.mean()
    return loss.item()

def evaluate_base(dataloader):
    data = torch.empty((0, 2048))

    with torch.no_grad():
        for samples in dataloader:
            data = torch.cat((data, samples), 0)
        mean = data.mean()
        mse = get_mse(data, mean)
    return mse

def inverse_standardize(raws, pred_un):
    mean = raws.mean(dim=1, keepdim=True)
    std = raws.std(dim=1, keepdim=True)
    return pred_un * std + mean

def inverse_log(raws, pred_un):
    # raws is not used, just to be consistent with other inverse functions
    return torch.exp(pred_un) - 1

def evaluate_inraw(model, dataloader_raw, dataloader_norm, inverse=None, device='cuda'):
    """
    dataloaders: the shuffling should be turned off for both dataloaders, otherwise the indices won't be matched.
                 the validation dataloader from get_dataloader() has shuffling turned off by default.
    inverse: function to inverse the normalized image to raw image
             inverse_standardize or inverse_log
    """
    total_loss = 0.
    model.eval()  # turn on evaluation mode

    with torch.no_grad():
        for (raws, norms) in zip(dataloader_raw, dataloader_norm):
            raws = raws.to(device, non_blocking=True, dtype=torch.float)
            norms = norms.to(device, non_blocking=True, dtype=torch.float)

            _, pred, _ = model(norms)    # in normalized space
            pred_un = model.unpatchify(pred) # the mae model has unpatchify function 
            if inverse:
                pred_un = inverse(raws, pred_un)    # in raw space
            loss = get_mse(raws, pred_un)
            total_loss += loss
    return total_loss / len(dataloader_raw)