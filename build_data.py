"""Build the Pytorch required data from the raw spectra.

It follows the workflow in pilot_01.ipynb to build the whole dataset.
"""

# get today's date
from datetime import date
date = date.today().strftime('%Y%m%d')


# create the directory for the spectra
import os
os.makedirs('data/spe', exist_ok=True)

# load the raw spectra with info
import pandas as pd
spe_df = pd.read_csv('data/spe_dataset_20220629.csv', index_col=0)

# filter out test set cores
cores = ['PS75-056-1', 'LV28-44-3', 'SO264-69-2']
spe_df = spe_df[~spe_df.core.isin(cores)].copy()

# set composite_id as a column
spe_df = spe_df.reset_index(drop=False)
print(spe_df.head())

# output the raw spectra
for row in spe_df.iterrows():
    row[1][1:2049].to_csv('data/pretrain/{}.csv'.format(row[0]), index=False, header=False)
print(f'{len(spe_df)} spectra exorted.')

# output the annotation file
spe_df['dirname'] = ['{}.csv'.format(id) for id in spe_df.index]
spe_df = spe_df[['dirname', 'composite_id', 'cps', 'core', 
                 'composite_depth_mm', 'section_depth_mm', 
                 'filename', 'section']]
spe_df.to_csv('data/spe_{}.csv'.format(date), index=False)
print(spe_df.head())