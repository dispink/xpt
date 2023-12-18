import torch

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