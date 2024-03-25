from torch.utils.tensorboard import SummaryWriter

def get_log_writer(args):
    return SummaryWriter(args.output_dir)