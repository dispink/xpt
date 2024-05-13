from argparse import ArgumentParser, Namespace


def get_train_args() -> Namespace:
    parser = ArgumentParser()
    # General parameters
    parser.add_argument("--annotation_file")
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Device parameters
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_memory", action="store_true")

    # Model parameters
    parser.add_argument("--model", default="base", choices=["base", "large", "huge"])
    parser.add_argument("--mask_ratio", default=0.8, type=float)

    # Hyper-parameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--transform", default='instance_normalize')

    # Optimizer parameters
    parser.add_argument("--optim", default="AdamW")
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--blr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=0, type=float)
    parser.add_argument("--betas", default=(0.9, 0.95), type=float, nargs='*')
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--lr_clip", default=0.5, type=float)

    # Learning rate scheduler parameters
    parser.add_argument("--lr_scheduler", default="warmup-cosine-annealing")
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--warmup_start_factor", default=0.5, type=float)
    parser.add_argument("--annealing_epochs", default=70)

    return parser.parse_args()


def get_tune_args() -> Namespace:
    parser = ArgumentParser()
    # General parameters
    parser.add_argument("--annotation_file")
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--verbose", action="store_true")

    # Device parameters
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_memory", action="store_true")

    # Model parameters
    parser.add_argument("--model", default="base", choices=["base", "large", "huge"])

    # Fine-tune parameters
    parser.add_argument("--pretrained_weight")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument(
        "--predict_layer", default="cls_token", choices=["cls_token", "global_pool"]
    )

    # Hyper-parameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--transform", default="instance_normalize")
    parser.add_argument("--transform_targets", default="instance_normalize")

    # Optimizer parameters
    parser.add_argument("--optim", default="AdamW")
    parser.add_argument("--lr_clip", default=0.5, type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--blr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=0, type=float)
    parser.add_argument("--betas", default=(0.9, 0.95), type=float, nargs='*')
    parser.add_argument("--layer_decay", type=float)
    parser.add_argument("--accum_iter", type=int)

    # Learning rate scheduler parameters
    parser.add_argument("--lr_scheduler", default="warmup-cosine-annealing")
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--warmup_start_factor", default=0.5, type=float)
    parser.add_argument("--annealing_epochs", default=70)

    return parser.parse_args()


def get_eval_args() -> Namespace:
    parser = ArgumentParser()
    # General parameters
    parser.add_argument("--pretrain")
    parser.add_argument("--downstream")
    parser.add_argument("--data_path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--log_dir")

    # Device parameters
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_memory", action="store_true")

    # Inference parameters
    parser.add_argument("--batch_size", ytpe=int)
    return parser.add_argument()
