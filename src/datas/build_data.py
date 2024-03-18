"""Prepare data ready for training from the raw spectra.
"""

import os
import pandas as pd


def split_sets(df, test_cores=["PS75-056-1", "LV28-44-3", "SO264-69-2"]):
    """
    Separate train+validation and test set cores, and
    return a dictionary with pd.DataFrames named as 'train' and 'test' keys.

    df: a pd.DataFrame of spectra or targets compiled
    test_cores: a list of test set core names, default is ['PS75-056-1', 'LV28-44-3', 'SO264-69-2'],
                which are the cores used in the ref paper, Lee et al. (2022).
    """
    data = {}
    data["train"] = df[~df.core.isin(test_cores)].copy()
    data["test"] = df[df.core.isin(test_cores)].copy()

    return data


def build_pretrain(out_dir: str, spe_csv: str):
    """Prepare the pretrain data ready for training.
    Args:
        out_dir: str, the directory to save the pretrain data.
        spe_csv: str, the path to the raw spectra csv.
    """

    # create the directory for the spectra
    os.makedirs(f"{out_dir}/train/spe", exist_ok=True)
    os.makedirs(f"{out_dir}/test/spe", exist_ok=True)

    # load the raw spectra with info
    spe_df = pd.read_csv(spe_csv, index_col=0)

    # set composite_id as a column
    spe_df = spe_df.reset_index(drop=False)
    print(spe_df.head())

    # split training and test set cores
    dfs = split_sets(spe_df)
    del spe_df

    # output files
    for dataset in ["train", "test"]:
        # raw spectra
        for row in dfs[dataset].iterrows():
            row[1][1:2049].to_csv(
                f"{out_dir}/{dataset}/spe/{row[0]}.csv", index=False, header=False
            )

        print(f"{len(dfs[dataset])} spectra exorted as {dataset} set.")

        # the annotation file
        dfs[dataset]["dirname"] = [f"{id}.csv" for id in dfs[dataset].index]
        dfs[dataset] = dfs[dataset][
            [
                "dirname",
                "composite_id",
                "cps",
                "core",
                "composite_depth_mm",
                "section_depth_mm",
                "filename",
                "section",
            ]
        ]
        dfs[dataset].to_csv(f"{out_dir}/{dataset}/info.csv", index=False)

        print(dfs[dataset].head())


# def build_finetune(out_dir: str, spe_csv: str, test_cores: list):

if __name__ == "__main__":
    out_dir = "data/pretrain"
    spe_csv = "data/legacy/spe_dataset_20220629.csv"
    build_pretrain(out_dir, spe_csv)
