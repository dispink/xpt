import os

import torch

from src.train import train
from src.datas import transforms
from src.datas.dataloader import get_dataloader
from src.models import mae_vit_regressor
from src.utils.args import get_tune_args
from src.utils.logging import get_log_writer
from src.utils.optim import get_optimizer_lr_scheduler


def main(args):
    log_writer = get_log_writer(args)

    if args.target_transform == "normalize":
        target_mean = torch.load(args.target_mean)
        target_std = torch.load(args.target_std)
        target_transform = transforms.Normalize(target_mean, target_std)
    elif args.target_transform == "instance_normalize":
        def target_transform(x): return x
    else:
        raise NotImplementedError

    if args.transform == "instance_normalize":
        dataloader = get_dataloader(
            ispretrain=False,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            transform=transforms.InstanceNorm(),
            target_transform=target_transform,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    elif args.transform == "normalize":
        norm_mean = torch.Tensor(torch.load('src/datas/xpt_spe_mean.pth'))
        norm_std = torch.Tensor(torch.load('src/datas/xpt_spe_std.pth'))
        dataloader = get_dataloader(
            ispretrain=False,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            transform=transforms.Normalize(norm_mean, norm_std),
            target_transform=target_transform,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    elif args.transform == "log":
        dataloader = get_dataloader(
            ispretrain=False,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=args.batch_size,
            transform=transforms.LogTransform(),
            target_transform=target_transform,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    else:
        raise NotImplementedError

    if args.model == "base":
        model = mae_vit_regressor.mae_vit_base_patch16(pretrained=not args.from_scratch,
                                                       weights=args.pretrained_weight,)
    criterion = torch.nn.MSELoss()

    optimizer, scheduler = get_optimizer_lr_scheduler(
        model.parameters(), args)

    train.finetune_trainer(model, criterion, dataloader, optimizer, scheduler,
                           args.epochs, log_writer, args)

    model = model.cpu()
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'model.ckpt'))


if __name__ == '__main__':
    args = get_tune_args()
    main(args)
