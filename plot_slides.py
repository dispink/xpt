import matplotlib.pyplot as plt
import numpy as np
import torch
from util.datasets import CustomImageDataset, split, standardize
from models_mae import mae_vit_base_patch16

def unpatchify(mask, pred, model):
    pred_un = model.unpatchify(pred)
    pred_un_arr = pred_un.squeeze(0).cpu().numpy()

    mask_arr = mask.squeeze(0).cpu().numpy()
    mask_un_arr = np.array([])
    for i in mask_arr:
        mask_un_arr = np.concatenate((mask_un_arr, np.repeat(i, 16)))
    mask_un_arr = mask_un_arr.astype(int)

    return pred_un_arr, mask_un_arr

dataset = CustomImageDataset(
    annotations_file='data/info_20231225.csv', input_dir='data/pretrain', 
    transform=standardize
    )
_, data_val = split(dataset)

model = mae_vit_base_patch16().to('cuda')
model.load_state_dict(torch.load('models/mae_vit_base_patch16_l-coslr_1e-05_20231229.pth'))

model.eval()
with torch.no_grad():
    spe_arr = data_val[22]
    spe = torch.tensor(spe_arr).unsqueeze(0).to('cuda', non_blocking=True, dtype=torch.float)
    _, pred, mask = model(spe)
    pred_un_arr, mask_un_arr = unpatchify(mask, pred, model)
    
# create figures with transparent background
channel = np.arange(1, len(spe_arr)+1)
#ylim = (-1, 11.8)

fig = plt.figure(figsize=(7, 5))
plt.plot(channel, spe_arr, alpha=.8, label='target', c='C0')
plt.vlines(channel, ymin=-.5, ymax=mask_un_arr*(spe_arr.max()), color='white', label='masked')
#plt.ylim(ylim)
plt.xlabel('Channel')
plt.ylabel('Standardized intensity')
plt.tight_layout()
plt.savefig('results/spe_with_mask.png', transparent=True)

fig = plt.figure(figsize=(7, 5))
plt.plot(channel, spe_arr, alpha=.8, label='target', c='C0')
#plt.ylim(ylim)
plt.xlabel('Channel')
plt.ylabel('Standardized intensity')
plt.tight_layout()
plt.savefig('results/spe_without_mask.png', transparent=True)

fig = plt.figure(figsize=(7, 5))
plt.plot(channel, pred_un_arr, alpha=.8, label='target', c='C1')
#plt.ylim(ylim)
plt.xlabel('Channel')
plt.ylabel('Standardized intensity')
plt.tight_layout()
plt.savefig('results/pred_without_mask.png', transparent=True)

#plt.figure(figsize=(7, 5))
#plt.plot(pred_un_arr)
#plt.plot(spe_arr)
#plt.savefig('results/pred.png')