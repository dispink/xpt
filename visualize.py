import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["lines.linewidth"] = 0.6
plt.rcParams["lines.markersize"] = 2
plt.rcParams["figure.autolayout"] = True


def demo_patch():
    from matplotlib.patches import Rectangle
    from src.datas import datasets, transforms

    # load dataset
    transform = transforms.InstanceNorm()
    dataset = datasets.PretrainDataset(
        annotations_file="data/pretrain/train/info.csv",
        input_dir="data/pretrain/train/",
        transform=transform
    )

    # tensor to numpy
    spe = dataset[0].numpy()
    patch_num = 10
    i = 0

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(spe)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Standardized intensity")
    rec = Rectangle((i * 16, -0.2), 16 * patch_num, 2, alpha=0.3, fill=None)
    ax.add_patch(rec)
    fig.savefig("files/demo_patch_1.png")

    fig, axes = plt.subplots(1, patch_num, figsize=(12, 3), sharey="row")
    for ax in axes:
        start = i * 16
        end = (i + 1) * 16
        ax.plot(range(start, end), spe[start:end], linewidth=2)
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

    # keep the subplots' background but remove the figure's background
    fig.savefig("files/demo_patch_2.png", transparent=True)


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


def check_transform():
    import torch
    from src.datas import datasets, transforms, dataloader
    from src.models import mae_vit

    mask_ratio = 0.5
    weights = f"results/HPtuning-loss-on-masks/pretrain-mask-ratio-0.5-blr-1e-4-transform-instance_normalize/model.ckpt"
    # size = 5  # number of spectra to plot, should be odd numbers
    size = [2, 2]
    transform = transforms.InstanceNorm()

    # load dataset
    dataset = datasets.PretrainDataset(
        annotations_file="data/pretrain/train/info.csv",
        input_dir="data/pretrain/train",
        transform=transform
    )

    _, data_val = dataloader.split(dataset)

    # load model
    model = mae_vit.mae_vit_base_patch16(mask_ratio=mask_ratio).to("cuda")
    model.load_state_dict(
        torch.load(weights)
    )

    model.eval()

    fig, axes = plt.subplots(
        size[0], size[1], sharex="col", sharey="all", figsize=(7.25, 4))
    for ax, index in zip(axes.ravel(), np.random.randint(0, 350, size[0]*size[1])):
        with torch.no_grad():
            spe = data_val[index].unsqueeze(0).to(
                "cuda", non_blocking=True, dtype=torch.float)
            _, pred, mask = model(spe, mask_only=True)
            pred_un_arr, mask_un_arr = unpatchify_PredandMask(
                mask, pred, model)

        spe_arr = spe.squeeze(0).cpu().numpy()

        # create figures with transparent background
        channel = np.arange(1, spe.shape[1] + 1)
        kev = channel * 0.02  # 20 eV/channel
        # ylim = (-4, spe_arr.max() + 1)
        ylim = (-0.8, 14.6)
        # 1 masked, 0 unmasked
        mask_un = (mask_un_arr == 1)

        # plot the original spectrum
        ax.plot(kev, spe_arr, alpha=0.9, label="original spectrum",
                c="C0", linewidth=0.4*size[0], zorder=5)
        ax.set_ylim(ylim)

        # plot the masked part
        ax.vlines(
            kev[mask_un],
            ymin=ylim[0],
            ymax=np.repeat(ylim[1], mask_un.sum()),
            color="#D3D3D3",
            label="masked part",
            zorder=0
            # alpha=0.1
        )

        # plot the reconstructed spectrum
        ax.scatter(kev[mask_un], pred_un_arr[mask_un], alpha=0.6,
                   label="reconstruction", c="black", s=1/(size[0]*size[1]), zorder=10)

    for i in range(size[0]):
        axes[i, 0].set_ylabel("Normalized intensity")
        axes[1, i].set_xlabel("Energy (KeV)")

    axes[1, 1].legend()

    fig.savefig(
        f"files/compiled_spectra.png")


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
    ax.plot(epochs, r2_pretrain, label="Pre-training", marker="x")
    ax.plot(epochs, r2_avg, label="Fine-tuning",
            c="gray", marker="o", markerfacecolor="none")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R$^2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("files/overfitting_in_pretrain.png", dpi=300)


def performance_mask_ratio_val():
    """Codes adopted from finetune_05.ipynb"""
    import pandas as pd
    styles = dict(
        marker=["x", "x", "x"],
        linestyle=["-", "--", ":"],
        c=["C0", "gray", "gray"]
    )

    r2_mean_df = pd.read_csv(
        "files/finetune_pretrained_r2_mean.csv", index_col=0)
    transforms = ["instance_normalize", "normalize", "log"]

    # 3-1 plot: r2_mean vs mask_ratio in each transform
    fig, ax = plt.subplots(figsize=(6, 3))

    for i, t in enumerate(transforms):
        mask = r2_mean_df["transform"] == t
        r2_mean_ratios = r2_mean_df[mask].groupby("mask_ratio").apply(
            lambda x: x["r2_mean"].max(), include_groups=False).copy()
        plt.plot(
            r2_mean_ratios.index, r2_mean_ratios, label=t,
            marker=styles["marker"][i], linestyle=styles["linestyle"][i],
            c=styles["c"][i], alpha=0.7)

    # ax.set_ylim(0.93, 0.98)
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

    # ax.set_ylim(0.966, 0.975)
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel("Avg. R$^2$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("files/r2_mean_vs_mask_ratio.png", dpi=300)


def detailed_performance_mask_ratio_val():
    fig, axes = plt.subplots(3, 1, sharex="col", figsize=(3.54, 5.6))

    # 3-3 plot
    r2_mean_df = pd.read_csv(
        "files/finetune_pretrained_r2_mean.csv", index_col=0)
    r2_mean_ratios = r2_mean_df.groupby("mask_ratio").apply(
        lambda x: x["r2_mean"].max(), include_groups=False).copy()
    axes[0].plot(r2_mean_ratios.index, r2_mean_ratios,
                 label="optimal model", marker="x", alpha=0.7)

    # 1-1 plots
    df = pd.read_csv("files/finetune_pretrained.csv", index_col=0)

    styles = dict(
        marker=["x", "x", "x"],
        linestyle=["-", "--", ":"],
        c=["C0", "gray", "gray"]
    )

    for ax, target in zip(axes[1:], ["TOC", "CaCO3"]):
        for i, t in enumerate(df["transform"].unique()):
            mask = (df["transform"] == t) & (df["target"] == target)
            best_r2_df = df[mask].groupby("mask_ratio").apply(
                lambda x: x.loc[x["r_square"].idxmax()], include_groups=False).copy()
            ax.plot(
                best_r2_df.index, best_r2_df["r_square"],
                label=t, marker=styles["marker"][i], linestyle=styles["linestyle"][i],
                c=styles["c"][i], alpha=0.9)

    ylabels = ["Avg. R$^2$", "R$^2$ (TOC)", "R$^2$ (CaCO$_3$)"]
    indices = ["a", "b", "c"]
    ylims = [(0.897, 0.9352), (0.49, 1.04), (0.49, 1.04)]
    legend_locs = ["upper right", "lower right", "lower right"]

    for i, ax in enumerate(axes):
        ax.text(0.01, 0.92,
                indices[i], transform=ax.transAxes, fontsize=9, weight='bold')
        ax.set_ylabel(ylabels[i])
        ax.set_ylim(ylims[i])
        ax.legend(loc=legend_locs[i])

    ax.set_xlabel("Mask Ratio")

    fig.savefig("files/r2_vs_mask_ratio_detailed.png")


def performance_data_val():
    """Codes adopted from finetune_data.ipynb"""
    import pandas as pd

    df = pd.read_csv("files/finetune_data_amount.csv", index_col=0)

    r2_2022 = {"CaCO3": 0.964, "TOC": 0.778}

    fig, axes = plt.subplots(1, 2, sharey="row", figsize=(3.54, 2.5))

    for target, ax, index in zip(["CaCO3", "TOC"], axes, ["a", "b"]):
        data_no = df.loc[df["target"] == target, "data_no"].values
        r2_ft = df.loc[df["target"] == target, "r2_ft"].values
        r2_scratch = df.loc[df["target"] == target, "r2_scratch"].values

        ax.plot(data_no, r2_ft, label="MAX", marker="x", alpha=0.7)
        ax.plot(data_no, r2_scratch, label="ViT-base",
                marker="x", ls="--", alpha=0.7, c="gray")
        ax.scatter(data_no[-1], r2_2022[target], marker="^",
                   label="baseline", alpha=0.7, c="black")

        ax.text(0.01, 0.92,
                index, transform=ax.transAxes, fontsize=9, weight='bold')
        ax.set_xlabel("Data amount")

        ax.set_xscale("log")

    axes[0].legend()
    axes[0].set_ylabel("R$^2$")
    fig.tight_layout()
    fig.savefig("files/r2_vs_data_amount.png")


def performance_data_case():
    df = pd.read_csv("files/finetune_data_amount_case.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(3.54, 2.1))

    ax.plot(
        df.loc[df["target"] == "CaCO3",
               "data_no"], df.loc[df["target"] == "CaCO3", "r2_ft"],
        label="CaCO$_3$", marker="x", alpha=0.7
    )
    ax.plot(
        df.loc[df["target"] == "TOC",
               "data_no"], df.loc[df["target"] == "TOC", "r2_ft"],
        label="TOC", marker="x", ls="--", c="gray", alpha=0.7
    )

    ax.set_xlabel("Data amount")

    ax.legend()
    ax.set_ylabel("R$^2$")
    plt.tight_layout()
    fig.savefig("files/r2_vs_data_amount_case.png", dpi=300)


def plot_ev_peaks(min: int, max: int, target, elements: list = False, legend: bool = True):
    "Modified from finetune_saliency_map_02.ipynb"
    from src.utils.saliency_map import filter_emission_peaks
    from src.utils.saliency_map import generate_saliency_map

    sa_array = generate_saliency_map(target)
    ev_compile_df = pd.read_csv("files/emission_peaks.csv", index_col=0)
    df = filter_emission_peaks(ev_compile_df, min, max)

    if elements:
        df = df[df["element"].isin(elements)].copy()
    start = int(min*50)
    end = int(max*50)
    ymax = 0.65

    fig = plt.figure(figsize=(7, 5))

    # draw soectrum
    plt.plot((np.linspace(1, 2048, 2048) * 0.02)
             [start:end], sa_array[start:end], c="gray")

    # draw ev peaks
    for i, row in enumerate(df.iterrows()):
        row = row[1][row[1] != ""].values
        plt.vlines(row[1:]*0.001, ymin=0, ymax=ymax,
                   label=row[0], colors=f"C{i}", alpha=0.5)

    plt.ylim(0, ymax)

    if legend:
        plt.legend()
    plt.xlabel("Energy (KeV)")
    plt.ylabel("Saliency")
    return fig


def customize_plot_ev_peaks(min: int, max: int, target, elements: list = False):
    fig = plot_ev_peaks(min, max, target, elements, legend=False)
    i = 0
    if target == "CaCO3":
        plt.vlines([7.38, 8.02], ymin=0, ymax=0.65,
                   label="Ca*2", colors="C3", alpha=0.5)
        for x, txt in zip([0.82, 2.2, 4.1, 6.7], ["Mg", "P", "Ca", "Ca*2"]):
            plt.text(x, 0.62, txt, fontsize=12, c=f"C{i}")
            i += 1
    elif target == "TOC":
        for x, txt in zip([1.2, 1.76, 3.2, 5.6], elements):
            plt.text(x, 0.62, txt, fontsize=12, c=f"C{i}")
            i += 1

    fig.savefig(f"results/saliency_map_{target}.png")


def combined_saliency_map():
    from src.utils.saliency_map import filter_emission_peaks, generate_saliency_map

    sa_dict = {
        "CaCO3": generate_saliency_map("CaCO3"),
        "TOC": generate_saliency_map("TOC")
    }

    elements_dict = {
        "CaCO3": ["Mg", "P", "Ca"],
        "TOC": ["Br", "Zr", "Rh", "Ba"]
    }

    min = 1
    max = 8.2
    ev_compile_df = pd.read_csv("files/emission_peaks.csv", index_col=0)
    df = filter_emission_peaks(ev_compile_df, min, max)
    start = int(min*50)
    end = int(max*50)
    ymin = -0.02
    ymax = 0.65
    kev = np.linspace(1, 2048, 2048) * 0.02
    figsize = (7, 3.7)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey="row", sharex="col")
    for ax_row, target in enumerate(["CaCO3", "TOC"]):
        sa_array = sa_dict[target]
        elements = elements_dict[target]

        label_y = 0.52
        i = 0
        if target == "CaCO3":
            axes[0, 1].vlines([7.38, 8.02], ymin=0,
                              ymax=0.65, colors="C3", alpha=0.5)
            for label_x, txt in zip([0.85, 2.2, 4.1, 6.75], ["Mg", "P", "Ca", "Ca*2"]):
                axes[0, 1].text(label_x, label_y, txt, fontsize=6, c=f"C{i}")
                i += 1

        elif target == "TOC":
            for label_x, txt in zip([1.15, 1.75, 3.2, 5.6], elements):
                axes[1, 1].text(label_x, label_y, txt, fontsize=6, c=f"C{i}")
                i += 1

        # draw the whole saliency map
        axes[ax_row, 0].plot(kev, sa_array, c="gray", label="whole map")

        # draw zoom-in saliency map
        axes[ax_row, 1].plot(
            kev[start:end], sa_array[start:end], c="gray", label="zoom-in map")

        df_tmp = df[df["element"].isin(elements)].copy()

        # draw ev peaks
        for i, row in enumerate(df_tmp.iterrows()):
            row = row[1][row[1] != ""].values
            axes[ax_row, 1].vlines(row[1:]*0.001, ymin=ymin, ymax=ymax,
                                   label=row[0], colors=f"C{i}", alpha=0.5)

        axes[ax_row, 1].set_ylim(ymin, ymax)

    # set axis labels
    axes[1, 0].set_xlabel("Energy (KeV) - whole")
    axes[1, 1].set_xlabel("Energy (KeV) - zoom-in")
    axes[0, 0].set_ylabel("Saliency - CaCO$_3$")
    axes[1, 0].set_ylabel("Saliency - TOC")

    # set axe indices
    for ax, index in zip(axes.ravel(), ["a", "b", "c", "d"]):
        ax.text(0.01, 0.92,
                index, transform=ax.transAxes, fontsize=9, weight='bold')

    fig.savefig("files/combined_saliency_map.png")


def plot_datasets():
    fig, axes = plt.subplots(
        2, 3, sharex='row', sharey='row', figsize=(7, 6.6))

    info_dict = {}
    for i, target in enumerate(["CaCO3%", "TOC%"]):
        info_dict["test"] = pd.read_csv(
            f"data/finetune/{target}/test/info.csv")
        info_dict["train"] = pd.read_csv(
            f"data/finetune/{target}/train/info.csv")
        info_dict["val"] = pd.read_csv(
            f"data/finetune/{target}/train/val.csv")

        for j, subset in enumerate(["train", "val", "test"]):
            y = []
            if subset == "test":
                for csv in info_dict[subset].dirname:
                    measurement = np.loadtxt(
                        f"data/finetune/{target}/test/target/{csv}", delimiter=",", dtype=float)
                    y.append(measurement)
            else:
                # train and val read the spe from the same directory but based on the different info.csv
                for csv in info_dict[subset].dirname:
                    measurement = np.loadtxt(
                        f"data/finetune/{target}/train/target/{csv}", delimiter=",", dtype=float)
                    y.append(measurement)
            y = np.array(y)

            axes[i, j].hist(y, bins=25, alpha=0.5)
            axes[i, j].text(0.62, 0.76, "max={:.1f}\nmin={:.2f}\nmean:{:.2f}\nstd={:.2f}\nN={}".format(
                y.max(), y.min(), y.mean(), y.std(), len(y)), transform=axes[i, j].transAxes, size=8
            )

    # The test set is named as case study after discussion
    for j, subset in enumerate(["Train", "Validation", "Case study"]):
        axes[0, j].set_title(subset, size=9)

    axes[0, 0].set_ylabel("Count for CaCO$_3$")
    axes[1, 0].set_ylabel("Count for TOC")
    axes[1, 1].set_xlabel("Concentration (wt%)")

    fig.savefig("files/data_hist.png")


if __name__ == "__main__":
    check_transform()
