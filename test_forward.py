import torch
from torch import Generator
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from util.datasets import CustomImageDataset
from models_mae import mae_vit_base_patch16

dataset = CustomImageDataset('data/info_20231121.csv', 'data/spe')
data_train, data_test = random_split(dataset, [0.8, 0.2], generator=Generator().manual_seed(24))

train_dataloader = DataLoader(data_train, batch_size=64, shuffle=True, )
batch = next(iter(train_dataloader))
print(batch.size())
print(batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mae_vit_base_patch16().to(device)

loss, _, _ = model(batch.to(device, dtype=torch.float))

print(loss)