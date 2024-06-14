import pandas as pd
from sklearn.model_selection import train_test_split

from src.datas.dataloader import split

target = 'TOC'
info_file = f'data/finetune/{target}%/train/info.csv'
total_df = pd.read_csv(info_file)
train, val = split(total_df)
train_df = total_df.iloc[train.indices]

for num in (10, 50, 100, 500, 1000, 1500):
    sub_train_df, _ = train_test_split(train_df, train_size=num)
    sub_train_df.to_csv(
        f'data/finetune/{target}%/train/info_train_{num}.csv',
        index=False
    )
