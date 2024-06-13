import os

import torch

from src.train import train
from src.datas import transforms
from src.datas.dataloader import get_dataloader
from src.models import mae_vit
from src.utils.args import get_train_args
from src.utils.logging import get_log_writer
from src.utils.optim import get_optimizer_lr_scheduler


def main(args):
    log_writer = get_log_writer(args)

    if args.transform == "instance_normalize":
        dataloader = get_dataloader(
            ispretrain=True,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            transform=transforms.InstanceNorm(),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    elif args.transform == "normalize":
        # TODO: calculate the mean and variance for each channel.
        norm_mean = torch.Tensor(torch.load('src/datas/xpt_spe_mean.pth'))
        norm_std = torch.Tensor(torch.load('src/datas/xpt_spe_std.pth'))
        dataloader = get_dataloader(
            ispretrain=True,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            transform=transforms.Normalize(norm_mean, norm_std),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    elif args.transform == "log":
        dataloader = get_dataloader(
            ispretrain=True,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            transform=transforms.LogTransform(),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    else:
        raise NotImplementedError

    if args.model == "base":
        model = mae_vit.mae_vit_base_patch16(mask_ratio=args.mask_ratio)
    elif args.model == "large":
        model = mae_vit.mae_vit_large_patch16(mask_ratio=args.mask_ratio)
    elif args.model == "huge":
        model = mae_vit.mae_vit_huge_patch14(mask_ratio=args.mask_ratio)

    optimizer, scheduler = get_optimizer_lr_scheduler(
        model.parameters(), args)

    train.trainer(model, dataloader, optimizer, scheduler,
                  args.epochs, log_writer, args)

    model = model.cpu()
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'model.ckpt'))


if __name__ == '__main__':
    args = get_train_args()
    main(args)
