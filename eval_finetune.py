from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.eval.eval import finetune_evaluate, finetune_evaluate_base
from src.models.mae_vit_regressor import mae_vit_base_patch16
from src.datas import transforms
from src.datas.dataloader import get_dataloader


def main(args):
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda')

    annotation_file = f'data/finetune/{args.target}%/train/info.csv'
    target_mean = torch.load(f"src/datas/xpt_{args.target}_target_mean.pth")
    target_std = torch.load(f"src/datas/xpt_{args.target}_target_std.pth")
    target_transform = transforms.Normalize(target_mean, target_std)

    model = mae_vit_base_patch16(pretrained=True,
                                 weights=args.weights).to(device)

    dataloader = get_dataloader(ispretrain=False,
                                annotations_file=annotation_file,
                                input_dir=f"data/finetune/{args.target}%/train",
                                batch_size=256,
                                transform=transforms.InstanceNorm(),
                                target_transform=target_transform,
                                num_workers=8)
    dataloader = dataloader['val']

    model_mse = finetune_evaluate(model=model,
                                  dataloader=dataloader,
                                  criterion=criterion)

    base_mse = finetune_evaluate_base(dataloader=dataloader,
                                      criterion=criterion,
                                      mean=target_mean)

    r_square = 1 - model_mse / base_mse

    print(
        f'MSE: {model_mse:.3f},'
        f' MSE of base model: {base_mse:.3f},'
        f' R2: {r_square:.3f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target', default='CaCO3')
    parser.add_argument('--weights')
    main(parser.parse_args())
