from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def get_log_writer(args):
    w = SummaryWriter(args.output_dir)
    hparam_dict = vars(args)
    hparam_dict['start-time'] = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    w.add_hparams(hparam_dict, {})
    return w
