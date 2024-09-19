import pandas as pd
import numpy as np


def filter_emission_peaks(df, min, max):
    mask = (df.iloc[:, 2:] > min*1000) & (df.iloc[:, 2:] < max*1000)
    tmp = df[mask].loc[mask.any(axis=1)].copy()
    tmp = tmp.replace(np.nan, "")
    element = df.loc[mask.any(axis=1), "element"].copy()
    return pd.concat([element, tmp.iloc[:, 2:]], axis=1)


def generate_saliency_map(target_task,
                          finetuned_weight_dir="results/finetune_pretrained/pretrain-mask-ratio-0.5-blr-1e-4-transform-instance_normalize"):
    """
    Return the mean aliency map of the last batch in np.array format. 
    """
    import torch
    from src.datas import transforms
    from src.datas.dataloader import get_dataloader
    from src.models import mae_vit_regressor

    annotation_file = f"data/finetune/{target_task}%/train/info.csv"
    input_dir = f"data/finetune/{target_task}%/train/"

    target_mean = torch.load(f"src/datas/xpt_{target_task}_target_mean.pth")
    target_std = torch.load(f"src/datas/xpt_{target_task}_target_std.pth")

    finetuned_weight = f"{finetuned_weight_dir}/{target_task}/model.ckpt"
    device = "cuda"

    target_transform = transforms.Normalize(target_mean, target_std)

    dataloader = get_dataloader(
        ispretrain=False,
        annotations_file=annotation_file,
        input_dir=input_dir,
        batch_size=56,
        transform=transforms.InstanceNorm(),
        target_transform=target_transform,
        num_workers=1,
        pin_memory=True,
    )

    model = mae_vit_regressor.mae_vit_base_patch16(pretrained=True,
                                                   weights=finetuned_weight)
    criterion = torch.nn.MSELoss()

    model.eval()
    model = model.cuda()

    # run the batches until the last batch
    for batch in dataloader["train"]:
        samples = batch["spe"].to(device, non_blocking=True, dtype=torch.float)
        samples.requires_grad = True
        targets = batch["target"].to(
            device, non_blocking=True, dtype=torch.float)

        preds = model(samples)
        loss = criterion(preds, targets)
        loss.backward()

        saliency_map = samples.grad.data.abs()
        saliency_map /= saliency_map.max(dim=-1)[0].unsqueeze(dim=-1)
    return saliency_map.cpu().numpy().mean(axis=0)
