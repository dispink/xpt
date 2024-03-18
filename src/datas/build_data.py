"""Prepare data ready for training from the raw spectra.
"""

import os
import pandas as pd


def build_pretrain(out_dir: str, spe_csv: str, test_cores: list):
    """Prepare the pretrain data ready for training.
    Args:
        out_dir: str, the directory to save the pretrain data.
        spe_csv: str, the path to the raw spectra csv.
        test_cores: list of test core names.
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
    train_df = spe_df[~spe_df.core.isin(test_cores)].copy()
    test_df = spe_df[spe_df.core.isin(test_cores)].copy()
    del spe_df

    # output files
    for df, dataset in zip([train_df, test_df], ["train", "test"]):
        # raw spectra
        for row in df.iterrows():
            row[1][1:2049].to_csv(
                f"{out_dir}/{dataset}/spe/{row[0]}.csv", index=False, header=False
            )

        print(f"{len(df)} spectra exorted as {dataset} set.")

        # the annotation file
        df["dirname"] = [f"{id}.csv" for id in df.index]
        df = df[
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
        df.to_csv(f"{out_dir}/{dataset}/info_{dataset}.csv", index=False)

        print(df.head())


if __name__ == "__main__":
    out_dir = "data/pretrain"
    spe_csv = "data/legacy/spe_dataset_20220629.csv"
    test_cores = ["PS75-056-1", "LV28-44-3", "SO264-69-2"]
    build_pretrain(out_dir, spe_csv, test_cores)
