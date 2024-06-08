from torch.utils.tensorboard import SummaryWriter


def get_log_writer(args):
    w = SummaryWriter(args.output_dir)
    w.add_hparams(vars(args), {})
    return w
