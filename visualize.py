import matplotlib.pyplot as plt
import os
import numpy as np


def demo_patch(root: str = os.getcwd()):
    from matplotlib.patches import Rectangle
    from src.datas import datasets, transforms

    # load dataset
    dataset = datasets.PretrainDataset(
        annotations_file=f"{root}/data/pretrain/train/info.csv",
        input_dir=f"{root}/data/pretrain/train/spe",
    )

    # tensor to numpy
    spe = transforms.standardize_numpy(dataset[0].numpy())
    patch_num = 10
    i = 0

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(spe)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Standardized intensity")
    rec = Rectangle((i * 16, -0.2), 16 * patch_num, 2, alpha=0.3, fill=None)
    ax.add_patch(rec)
    fig.tight_layout()
    fig.savefig(f"{root}/results/demo_patch_1.png")

    fig, axes = plt.subplots(1, patch_num, figsize=(12, 3), sharey="row")
    for ax in axes:
        start = i * 16
        end = (i + 1) * 16
        ax.plot(range(start, end), spe[start:end])
        # ax.axis("off")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        # ax.set_facecolor("white")
        i += 1

    fig.tight_layout()

    # keep the subplots' background but remove the figure's background
    fig.savefig(f"{root}/results/demo_patch_2.png", transparent=True)


def unpatchify_PredandMask(mask, pred, model):
    pred_un = model.unpatchify(pred)
    pred_un_arr = pred_un.squeeze(0).cpu().numpy()

    mask_arr = mask.squeeze(0).cpu().numpy()
    mask_un_arr = np.array([])
    for i in mask_arr:
        mask_un_arr = np.concatenate((mask_un_arr, np.repeat(i, 16)))
    mask_un_arr = mask_un_arr.astype(int)

    return pred_un_arr, mask_un_arr


def demo_reconstruction(root: str = os.getcwd()):
    import torch
    from src.datas import datasets, transforms, dataloader
    from src.models import mae_vit

    # load dataset

    dataset = datasets.PretrainDataset(
        annotations_file=f"{root}/data/pretrain/train/info.csv",
        input_dir=f"{root}/data/pretrain/train/spe",
    )

    _, data_val = dataloader.split(dataset)

    print(data_val[22])

    model = mae_vit.mae_vit_base_patch16().to("cuda")
    model.load_state_dict(
        torch.load("models/mae_vit_base_patch16_l-coslr_1e-05_20231229.pth")
    )

    model.eval()
    with torch.no_grad():
        spe_arr = transforms.standardize_numpy(data_val[22].numpy())
        spe = (
            torch.tensor(spe_arr)
            .unsqueeze(0)
            .to("cuda", non_blocking=True, dtype=torch.float)
        )
        _, pred, mask = model(spe)
        pred_un_arr, mask_un_arr = unpatchify_PredandMask(mask, pred, model)

    # create figures with transparent background
    channel = np.arange(1, len(spe_arr) + 1)
    ylim = (-1, 11.8)

    # plot the masked spectrum
    fig = plt.figure(figsize=(7, 5))
    plt.plot(channel, spe_arr, alpha=0.8, label="target", c="C0")
    plt.ylim(ylim)
    plt.vlines(
        channel,
        ymin=-0.5,
        ymax=mask_un_arr * (spe_arr.max()),
        color="white",
        label="masked",
    )
    plt.xlabel("Channel")
    plt.ylabel("Standardized intensity")
    plt.tight_layout()
    plt.savefig(f"{root}/results/spe_with_mask.png", transparent=True)

    # plot the unmasked spectrum
    fig = plt.figure(figsize=(7, 5))
    plt.plot(channel, spe_arr, alpha=0.8, label="target", c="C0")
    plt.ylim(ylim)
    plt.xlabel("Channel")
    plt.ylabel("Standardized intensity")
    plt.tight_layout()
    plt.savefig(f"{root}/results/spe_without_mask.png", transparent=True)

    # plot the predicted spectrum
    fig = plt.figure(figsize=(7, 5))
    plt.plot(channel, pred_un_arr, alpha=0.8, label="target", c="C1")
    plt.ylim(ylim)
    plt.xlabel("Channel")
    plt.ylabel("Standardized intensity")
    plt.tight_layout()
    plt.savefig(f"{root}/results/pred_without_mask.png", transparent=True)


def check_transform(mask_ratio, weights, root: str = os.getcwd(), transform="normalize"):
    import torch
    from src.datas import datasets, transforms, dataloader
    from src.models import mae_vit

    if transform == "instance_normalize":
        transform = transforms.InstanceNorm()

    elif transform == "normalize":
        norm_mean = torch.Tensor(torch.load('src/datas/xpt_spe_mean.pth'))
        norm_std = torch.Tensor(torch.load('src/datas/xpt_spe_std.pth'))
        transform = transforms.Normalize(norm_mean, norm_std)

    elif transform == "log":
        transform = transforms.LogTransform()

    # load dataset
    dataset = datasets.PretrainDataset(
        annotations_file=f"{root}/data/pretrain/train/info.csv",
        input_dir=f"{root}/data/pretrain/train",
        transform=transform
    )

    _, data_val = dataloader.split(dataset)

    # load model
    model = mae_vit.mae_vit_base_patch16(mask_ratio=mask_ratio).to("cuda")
    model.load_state_dict(
        torch.load(weights)
    )

    model.eval()
    with torch.no_grad():
        spe = data_val[22].unsqueeze(0).to(
            "cuda", non_blocking=True, dtype=torch.float)
        loss, pred, mask = model(spe)
        pred_un_arr, mask_un_arr = unpatchify_PredandMask(mask, pred, model)

    spe_arr = spe.squeeze(0).cpu().numpy()

    # create figures with transparent background
    channel = np.arange(1, spe.shape[1] + 1)
    ylim = (-4, spe_arr.max() + 1)
    # 1 masked, 0 unmasked
    mask_un = (mask_un_arr == 1)

    # plot the masked spectrum
    fig = plt.figure(figsize=(7, 5))
    plt.plot(channel, spe_arr, alpha=0.5, label="target", c="C0")
    plt.ylim(ylim)
    plt.vlines(
        channel[mask_un],
        ymin=ylim[0],
        ymax=np.repeat(ylim[1], mask_un.sum()),
        color="gray",
        label="masked",
        alpha=0.1
    )
    plt.plot(channel, pred_un_arr, alpha=0.5,
             label=f"pred (mse={loss:.2f})", c="C1")

    plt.xlabel("Channel")
    plt.ylabel("Standardized intensity")
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        f"{root}/results/spe_optimal-mask-ratio-{mask_ratio}.png")


def overfitting_in_pretrain():
    """
    Using different checkpoints in the pretraining to finetune on the task.
    Pretrained with: lr 1e-5, instance norm, epoch 100
    """
    epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mse_val = [0.0388, 0.0209, 0.0158, 0.0136,
               0.0102, 0.0096, 0.0070, 0.0066, 0.0057, 0.0054]
    mse_base_val = 0.99951
    r2_pretrain = 1 - (np.array(mse_val) / mse_base_val)
    r2_CaCO3 = [0.966, 0.966, 0.968, 0.970,
                0.971, 0.972, 0.971, 0.971, 0.971, 0.970]
    r2_TOC = [0.940, 0.909, 0.927, 0.943, 0.942,
              0.945, 0.947, 0.949, 0.946, 0.946]
    r2_avg = (np.array(r2_CaCO3) + np.array(r2_TOC)) / 2

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # plot the line with empty circle markers
    ax.plot(epochs, r2_pretrain , label="Pre-training", marker="x")
    ax.plot(epochs, r2_avg, label="Fine-tuning", c="gray", marker="o", markerfacecolor="none")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R$^2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("files/overfitting_in_pretrain.png", dpi=300)


def performance_mask_ratio_val():
    """Codes adopted from finetune_05.ipynb"""
    import pandas as pd

    df = pd.read_csv("files/finetune_pretrained.csv", index_col=0)
    transforms = ["instance_normalize", "normalize", "log"]

    # calculate the mean of r_square
    r2_mean_df = pd.DataFrame()
    for i in range(0, len(df), 2):
        r2_mean = df.loc[i:i+1, "r_square"].mean()
        tmp_df = df.iloc[i, :3].copy()
        tmp_df["r2_mean"] = r2_mean
        r2_mean_df = pd.concat([r2_mean_df, tmp_df], axis=1)

    r2_mean_df = r2_mean_df.T.reset_index(drop=True)

    # 3-1 plot: r2_mean vs mask_ratio in each transform
    fig, ax = plt.subplots(figsize=(6, 3))

    for t, marker in zip(transforms, ["x", "o", "^"]):
        mask = r2_mean_df["transform"] == t
        r2_mean_ratios = r2_mean_df[mask].groupby("mask_ratio").apply(
            lambda x: x["r2_mean"].max(), include_groups=False).copy()
        ax.plot(r2_mean_ratios.index, r2_mean_ratios,
                label=t, marker=marker, alpha=0.7)

    ax.set_ylim(0.93, 0.98)
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel("Avg. R$^2$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("files/r2_mean_vs_mask_ratio_norm.png", dpi=300)

    # 3-2 plot: r2_mean vs mask_ratio
    # This one I only plot the optmial model in each mask_ratio
    fig, ax = plt.subplots(figsize=(6, 3))

    r2_mean_ratios = r2_mean_df.groupby("mask_ratio").apply(
        lambda x: x["r2_mean"].max(), include_groups=False).copy()
    ax.plot(r2_mean_ratios.index, r2_mean_ratios,
            label="Optimal model", marker="x", alpha=0.7)

    ax.set_ylim(0.966, 0.975)
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel("Avg. R$^2$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("files/r2_mean_vs_mask_ratio.png", dpi=300)


def performance_data_val():
    """Codes adopted from finetune_data.ipynb"""
    import pandas as pd

    df = pd.read_csv("files/finetune_data_amount.csv", index_col=0)

    r2_2022 = {"CaCO3": 0.964, "TOC": 0.778}

    fig, axes = plt.subplots(1, 2, sharey="row", figsize=(10, 5))

    for target, ax in zip(["CaCO3", "TOC"], axes):
        data_no = df.loc[df["target"] == target, "data_no"].values
        r2_ft = df.loc[df["target"] == target, "r2_ft"].values
        r2_scratch = df.loc[df["target"] == target, "r2_scratch"].values

        ax.plot(data_no, r2_ft, label="ft", marker="x", alpha=0.7)
        ax.plot(data_no, r2_scratch, label="scratch",
                marker="x", ls="--", alpha=0.7, c="gray")
        ax.scatter(data_no[-1], r2_2022[target], marker="^",
                   label="conventional ML", alpha=0.7, c="black")

        ax.set_xlabel("Data amount")
        if target == "CaCO3":
            target = "CaCO$_3$"
        ax.set_title(target)
        ax.set_xscale("log")

    axes[1].legend()
    axes[0].set_ylabel("R$^2$")
    fig.tight_layout()
    fig.savefig("files/r2_vs_data_amount.png", dpi=300)


if __name__ == "__main__":
    overfitting_in_pretrain()
