"""Prepare data ready for training from the raw spectra.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_sets(df,
               train_cores=[
                   "LV29-114-3",
                   "SO178-12-3",
                   "SO264-13-2",
                   "SO264-15-2",
                   "SO264-28-2",
                   "SO264-55-1",
                   "SO264-56-2",
                   "SO264-64-1",
                   "SO264-66-2",
                   "PS97-027-2",
                   "PS97-046-4",
                   "PS97-052-4",
                   "PS97-053-2",
                   "PS97-078-1",
                   "PS97-079-2",
                   "PS97-080-1",
                   "PS97-083-2",
                   "PS97-084-1",
                   "PS97-085-3",
                   "PS97-089-1",
                   "PS97-092-1",
                   "PS97-093-2",
                   "PS97-128-2",
                   "PS75-054-1",
                   "PS75-083-1",
                   "PS75-093-1",
                   "PS75-095-5",
               ],
               test_cores=["PS75-056-1", "LV28-44-3", "SO264-69-2"]):
    """
    Separate train+validation and test set cores, and
    return a dictionary with pd.DataFrames named as 'train' and 'test' keys.

    df: a pd.DataFrame of spectra or targets compiled
    train_cores: a list of train set core names. 
    test_cores: a list of test set core names, default is ['PS75-056-1', 'LV28-44-3', 'SO264-69-2'],
                which are the cores used in the ref paper, Lee et al. (2022).
    """
    data = {}
    data["train"] = df[df.core.isin(train_cores)].copy()
    data["test"] = df[df.core.isin(test_cores)].copy()

    return data


# Data for pretrain
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


# Data for fine-tune
def replace_zero_and_negative(df, target):
    """
    Replace zero and negative values in targets with 0.001.
    """

    df[target] = df[target].apply(lambda x: 0.001 if x <= 0 else x)

    return df


def export_rows(df, out_dir, target):
    """
    Export the spectrum and target of each row in df to the "spe" and "target" folders under out_dir.
    In the meantime, return a list of filenames for later exporting an annotation file.

    df: a pd.DataFrame of spectra and desired target that don't have NaN
    out_dir: str, the directory to save the data.
    target: target name
    """
    filenames = []

    for i, row in enumerate(df.iterrows()):
        filename = f"{i}.csv"
        # export spectrum
        row[1][0:2048].to_csv(
            f"{out_dir}/spe/{filename}", index=False, header=False)
        # export target
        with open(f"{out_dir}/target/{filename}", "w") as f:
            f.write(str(row[1][target]))
        # prepare filename for the annotation file
        filenames.append(filename)
    return filenames


def export_data(df, out_dir, target, do_split=True):
    """
    Export the rows (spectrum and target) and the annotation file to the sub_dir.

    df: a pd.DataFrame of spectra and desired target
    out_dir: str, the directory to save the data.
    target: target name
    """

    df = df[~df[target].isna() & df[target] >= 0].copy()
    df["filename"] = export_rows(df, out_dir, target)
    print(f"{len(df)} data are exported.")

    # output the annotation file
    df = df[["filename", "core", "mid_depth_mm"]]

    # train validation splits
    if do_split:
        train_df, val_df = train_test_split(df, test_size=0.2)

        val_df.to_csv(f"{out_dir}/val.csv", index=False)

        train_df.to_csv(f"{out_dir}/info.csv", index=False)
        for size in [10, 50, 100, 500, 1000, 1500]:
            sub_df, _ = train_test_split(train_df, train_size=size)
            sub_df.to_csv(f"{out_dir}/info_{size}.csv", index=False)
    else:
        df.to_csv(f"{out_dir}/info.csv", index=False)


def build_finetune(out_dir: str, spe_csv: str, targets: list):
    # load the joined spe and targets files with info
    compile_df = pd.read_csv(spe_csv, index_col=0)

    # seperate train+validation and test set cores
    dfs = split_sets(compile_df)
    del compile_df

    for target in targets:
        for dataset in dfs.keys():
            df = replace_zero_and_negative(dfs[dataset], target)
            print(df[target].describe())

            # create subdirectories
            sub_dir = f"{out_dir}/{target}/{dataset}"
            os.makedirs(f"{sub_dir}/spe", exist_ok=True)
            os.makedirs(f"{sub_dir}/target", exist_ok=True)

            export_data(df, sub_dir, target, do_split=dataset == "train")


if __name__ == "__main__":
    build_pretrain(
        out_dir="data/_pretrain",
        spe_csv="data/legacy/spe_dataset_20220629.csv"
    )
    build_finetune(
        out_dir="data/_finetune",
        spe_csv="data/legacy/spe+bulk_dataset_20220629.csv",
        targets=["CaCO3%", "TOC%"]
    )
