from torch import optim


def get_optimizer(params, args):
    if args.optim == 'AdamW':
        optimizer = optim.AdamW(params, args.blr, betas=tuple(args.betas))
    else:
        raise NotImplementedError(f'{args.optim} is not implemented.')
    return optimizer


def get_lr_scheduler(optimizer, args):
    return


def get_optimizer_lr_scheduler(params, args):
    optimizer = get_optimizer(params, args)
    lr_scheduler = get_lr_scheduler(optimizer, args)
    return optimizer, lr_scheduler
