from argparse import ArgumentParser, Namespace


def get_train_parser() -> Namespace:
    parser = ArgumentParser()
    # General parameters
    parser.add_argument('--data_path')
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('--resume', action='store_true')

    # Device parameters
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='base',
                        choices=['base', 'large', 'huge'])
    parser.add_argument('--mask_ratio', default=0.8, type=float)

    # Hyper-parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--blr', type=float)
    parser.add_argument('--min_lr', type=float)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    return parser.parse_args()
