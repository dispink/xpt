#! usr/local/bin/python

def is_weight_equal():
    import torch
    a = torch.load(
        "results/pretrain-mask-ratio-0.3-blr-1e-6-transform-instance_normalize/model.ckpt")
    b = torch.load(
        "results/pretrain-mask-ratio-0.1-blr-1e-6-transform-instance_normalize/model.ckpt")

    for key in a.keys():
        if not torch.equal(a[key], b[key]):
            print(key)
            print(torch.equal(a[key], b[key]))
            print(torch.norm(a[key] - b[key]))
        else:
            print(key)
            print(torch.equal(a[key], b[key]))
            print("equal")


def count_models(dir):
    import glob

    models = glob.glob(f"{dir}/*/model.ckpt")
    print(len(models))


if __name__ == "__main__":
    count_models("results/HPtuning")
