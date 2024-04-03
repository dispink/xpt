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


if __name__ == "__main__":
    # get root directory
    demo_reconstruction()
