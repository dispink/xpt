"""Build the Pytorch required data for finetuning.

It modifies build_data.py for the downstream tasks: CaCO3 and TOC.
The input compiled dataset is spe+bulk_dataset_20220629.csv, in format:

     0    1    2    3    4    5    6    7    8    9   10   11  ...  2041  2042  2043  2044  2045  2046  2047       TC%      TOC%     CaCO3%        core  mid_depth_mm
0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0  2.542079  0.394127  17.898887  SO264-64-1         115.0
1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0  2.247150  0.611208  13.632300  SO264-64-1         215.0
2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.710588  0.523402   1.559822  SO264-64-1         305.0
3  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.562171  0.472551   0.746802  SO264-64-1        1015.0
4  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.578167  0.312852   2.210866  SO264-64-1        1815.0

Total length: 2643
TC length: 2196
TOC length: 2643
CaCO3 length: 2612

The output directory structure (data amount):
|-------main_dir|
|--------------------------sub_dir| 
                                  
                                   --info.csv
                --train (2103) ---|--spe
                |                  --target
finetune (2612)--
                |                  --info.csv
                --test (509) -----|--spe
                                   --target
  
"""

import os
import datetime
import pandas as pd

def get_date():
    return datetime.date.today().strftime("%Y%m%d")

def replace_zero_and_negative(df, targets):
    """
    Replace zero and negative values in targets with 0.001.
    """
    
    for target in targets:
        df[target] = df[target].apply(lambda x: 0.001 if x <= 0 else x)

    return df

def split_sets(df, test_cores=['PS75-056-1', 'LV28-44-3', 'SO264-69-2']):
    """
    Separate train+validation and test set cores, and 
    return a dictionary with pd.DataFrames named as 'train' and 'test' keys.

    df: a pd.DataFrame with spectra and targets compiled
    test_cores: a list of test set core names, default is ['PS75-056-1', 'LV28-44-3', 'SO264-69-2'],
                which are the cores used in the ref paper, Lee et al. (2022).
    """
    data = {}
    data['train'] = df[~df.core.isin(test_cores)].copy()
    data['test'] = df[df.core.isin(test_cores)].copy()

    return data

def export_rows(df, sub_dir, targets):
    """
    Export the spectrum and target of each row in df to the "spe" and "target" folders under sub_dir.
    In the meantime, return a list of filenames for later exporting an annotation file.

    df: a pd.DataFrame with spectra and desired targets that don't have NaN
    sub_dir: a string of the subdirectory, inherited from export_datasets()
    targets: a list of target names
    """
    filenames = []

    for row in df.iterrows():
        filename = f'{row[0]}.csv'
        # export spectrum
        row[1][0:2048].to_csv(f'{sub_dir}/spe/{filename}', index=False, header=False)
        # export target
        row[1][targets].to_csv(f'{sub_dir}/target/{filename}', index=False, header=False)
        # prepare filename for the annotation file
        filenames.append(filename)

    return filenames

def export_datasets(data, main_dir, targets=['CaCO3%', 'TOC%']):
    """
    Export the spectra and targets from data to the "train" and "test" folders under main_dir.

    data: a dictionary with keys 'train' and 'test' in pd.DataFrame format
    main_dir: a string of the main directory
    targets: a list of target names, default is ['CaCO3%', 'TOC%']
    """
    for dataset in data.keys():
        sub_dir = f'{main_dir}/{dataset}'

        # create subdirectories
        os.makedirs(f'{sub_dir}', exist_ok=True)
        os.makedirs(f'{sub_dir}/spe', exist_ok=True)
        os.makedirs(f'{sub_dir}/target', exist_ok=True)
        
        mask = ~data[dataset][targets].isna().any(axis=1)
        df = data[dataset][mask].copy()     
        df['filename'] = export_rows(df, sub_dir, targets)
        print(f'{len(df)} data for {dataset} are exported.')
        
        # output the annotation file
        df = df[['filename', 'core', 'mid_depth_mm']]
        df.to_csv(f'{sub_dir}/info_{get_date()}.csv', index=False)

def main():    
    # define the targets
    targets = ['CaCO3%', 'TOC%']

    # load the joined spe and targets files with info
    compile_df = pd.read_csv('data/spe+bulk_dataset_20220629.csv', index_col=0)

    # replace zero and negative values with 0.001
    compile_df = replace_zero_and_negative(compile_df, targets=targets)
    print(compile_df[targets].describe())

    # seperate train+validation and test set cores
    data = split_sets(compile_df)

    main_dir = 'data/finetune'
    # create the main directory
    os.makedirs(main_dir, exist_ok=True)

    # output the raw spectra and targets
    export_datasets(data, main_dir, targets=targets)

if __name__ == '__main__':
    main()