import matplotlib.pyplot as plt
import os


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


if __name__ == "__main__":
    # get root directory
    demo_patch()
