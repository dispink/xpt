#! /bin/bash

# Evaluate the optimal finetuned model on the test set

target=TOC
input_dir=data/finetune/${target}%/test
scale=instance_normalize
weights_dir=results/finetune_pretrained/pretrain-mask-ratio-0.9-blr-1e-4-transform-${scale}/${target}/model.ckpt

python eval_finetune.py \
    --target $target \
    --annotation_file ${input_dir}/info.csv \
    --input_dir $input_dir \
    --transform $scale \
    --weight $weights_dir \
    --test-only

# Results:
# CaCO3 MSE: 9.924,     MSE of base model: 79.463,      R2: 0.875
# TOC   MSE: 0.045,     MSE of base model: 0.195,       R2: 0.772