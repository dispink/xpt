import os

import torch
from torch import nn

from src.utils.args import get_eval_tune_args
from src.eval.eval import finetune_evaluate, finetune_evaluate_base
from src.models.mae_vit_regressor import mae_vit_base_patch16
from src.datas import transforms
from src.datas.dataloader import get_dataloader


class DeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return self.std * x + self.mean


def main(args):
    """
    No need to transform the target values because we wish to see the metrics in the original scale.
    Hence, the prediction of model needs to be denormalized back.
    """
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda')

    target_mean = torch.load(f"src/datas/xpt_{args.target}_target_mean.pth")
    target_std = torch.load(f"src/datas/xpt_{args.target}_target_std.pth")
    reverse_pred = DeNormalize(target_mean, target_std)

    model = mae_vit_base_patch16(pretrained=True,
                                 weights=args.weights).to(device)

    if args.transform == "instance_normalize":
        dataloader = get_dataloader(
            ispretrain=False,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=1,
            transform=transforms.InstanceNorm(),
            num_workers=8,
            test_only=args.test_only,
        )
    elif args.transform == "normalize":
        norm_mean = torch.Tensor(torch.load('src/datas/xpt_spe_mean.pth'))
        norm_std = torch.Tensor(torch.load('src/datas/xpt_spe_std.pth'))
        dataloader = get_dataloader(
            ispretrain=False,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=1,
            transform=transforms.Normalize(norm_mean, norm_std),
            num_workers=8,
            test_only=args.test_only,
        )
    elif args.transform == "log":
        dataloader = get_dataloader(
            ispretrain=False,
            annotations_file=args.annotation_file,
            input_dir=args.input_dir,
            batch_size=1,
            transform=transforms.LogTransform(),
            num_workers=8,
            test_only=args.test_only,
        )
    else:
        raise NotImplementedError

    if args.test_only:
        dataloader = dataloader['test']
    else:
        dataloader = dataloader['val']

    model_mse = finetune_evaluate(model=model,
                                  dataloader=dataloader,
                                  criterion=criterion,
                                  reverse_pred=reverse_pred)

    base_mse = finetune_evaluate_base(dataloader=dataloader,
                                      criterion=criterion,
                                      mean=target_mean)

    r_square = 1 - model_mse / base_mse

    results = f'MSE: {model_mse:.3f},\tMSE of base model: {base_mse:.3f},\tR2: {r_square:.3f}'
    print(results)

    # Save the results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f'{args.output_dir}/{args.target}.txt', 'w') as f:
            f.write(results)


if __name__ == '__main__':
    args = get_eval_tune_args()
    main(args)
