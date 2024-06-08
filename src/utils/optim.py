from torch import optim


def get_optimizer(params, args):
    if args.optim == "Adam":
        optimizer = optim.Adam(params, args.blr)
    elif args.optim == "AdamW":
        optimizer = optim.AdamW(
            params, args.blr, betas=(args.beta0, args.beta1))
    else:
        raise NotImplementedError(f"{args.optim} is not implemented.")
    return optimizer


def get_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "warmup-cosine-annealing":
        scheduler0 = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=args.warmup_start_factor, end_factor=1.0
        )
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler0, scheduler1],
            milestones=[args.warmup_epochs],
        )
    else:
        raise NotImplementedError(f"{args.lr_scheduler} is not implemented.")
    return scheduler


def get_optimizer_lr_scheduler(params, args):
    optimizer = get_optimizer(params, args)
    lr_scheduler = get_lr_scheduler(optimizer, args)
    return optimizer, lr_scheduler
