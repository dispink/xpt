import pandas as pd
from sklearn.model_selection import train_test_split

from src.datas.dataloader import split

target = 'TOC'

# the test set in the data directory means the case study
subset = 'test'

info_file = f'data/finetune/{target}%/{subset}/info.csv'
total_df = pd.read_csv(info_file)
train, val = split(total_df)

train_df = total_df.iloc[train.indices]
train_df.to_csv(
    f"data/finetune/{target}%/{subset}/info_train.csv", index=False)

val_df = total_df.iloc[val.indices]
val_df.to_csv(f"data/finetune/{target}%/{subset}/val.csv", index=False)

for num in (10, 50, 100, 150, 200, 250):
    sub_train_df, _ = train_test_split(train_df, train_size=num)
    sub_train_df.to_csv(
        f'data/finetune/{target}%/{subset}/info_{num}.csv',
        index=False
    )
