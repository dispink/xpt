import torch
import pandas as pd

from src.inference.inference import inference, DeNormalize
from src.models.mae_vit_regressor import mae_vit_base_patch16
from src.datas import transforms

from src.datas import transforms
from src.datas.dataloader import get_dataloader


def main():
    """
    We use the test cores in pretraining dataset because they are the same cores as the test cores in finetuning dataset,
    but with higher resolution and no target values.
    The target values require customized codes to make dataloader.
    Since we already have the target values in csv, so we skip the incovenience.
    """
    # We focus on the optimal fine-tuned model's zeroshot performance on the test set (aka case study).
    pretrain_blr = "1e-4"
    mask_ratio = "0.5"
    scale = "instance_normalize"
    annotation_file = f"data/pretrain/test/info.csv"
    input_dir = f"data/pretrain/test/"
    device = torch.device('cuda')

    if scale == "instance_normalize":
        transform = transforms.InstanceNorm()

    dataloader = get_dataloader(
        ispretrain=True,
        annotations_file=annotation_file,
        input_dir=input_dir,
        transform=transform,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        test_only=True
    )

    info_df = pd.read_csv(annotation_file)

    for target in ["CaCO3", "TOC"]:
        pretrained_weight = f"results/finetune_pretrained/pretrain-mask-ratio-{mask_ratio}-blr-{pretrain_blr}-transform-{scale}/{target}/model.ckpt"

        target_mean = torch.load(
            f"src/datas/xpt_{target}_target_mean.pth", weights_only=True)
        target_std = torch.load(
            f"src/datas/xpt_{target}_target_std.pth", weights_only=True)

        reverse_pred = DeNormalize(target_mean, target_std)

        model = mae_vit_base_patch16(pretrained=True,
                                     weights=pretrained_weight).to(device)

        predictions = inference(model, dataloader["test"], device=device)

        # denormalize the predictions
        predictions = reverse_pred(predictions).cpu().numpy().squeeze()

        info_df['{} prediction (wt%)'.format(target)] = predictions
    info_df.to_csv(f"results/zeroshot_case.csv")


if __name__ == '__main__':
    main()
