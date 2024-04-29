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
            transform=transforms.standardize_numpy,
            args=args,
        )
    elif args.transform == "normalize":
        # TODO: calculate the mean and variance for each channel.
        data_transformer = transforms.NormalizeTransform()
        dataloader = get_dataloader(
            ispretrain=True,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            data_transformer=data_transformer,
            args=args,
        )
    elif args.transform == "log":
        data_transformer = transforms.LogTransform()
        dataloader = get_dataloader(
            ispretrain=True,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            data_transformer=data_transformer,
            args=args,
        )
    else:
        raise NotImplementedError

    if args.model == "base":
        model = mae_vit.mae_vit_base_patch16()
    elif args.model == "large":
        model = mae_vit.mae_vit_large_patch16()
    elif args.model == "huge":
        model = mae_vit.mae_vit_huge_patch14()

    optimizer, scheduler = get_optimizer_lr_scheduler(
        model.parameters(), args)

    train.trainer(model, dataloader, optimizer, scheduler,
                  args.epochs, log_writer, args)

    model = model.cpu()
    torch.save(model.state_dict(),
               os.path.join(model.state_dict(), 'model.ckpt'))


if __name__ == '__main__':
    args = get_train_args()
    main(args)
