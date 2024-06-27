#! /bin/bash

# Evaluate the optimal finetuned model on the test set

target=CaCO3
input_dir=data/finetune/${target}%/test
pretrain_blr="1e-4"
mask_ratio="0.7"
scale=instance_normalize
weights_dir=results/finetune_pretrained/pretrain-mask-ratio-${mask_ratio}-blr-${pretrain_blr}-transform-${scale}/${target}/model.ckpt

python eval_finetune.py \
    --target $target \
    --annotation_file ${input_dir}/info.csv \
    --input_dir $input_dir \
    --transform $scale \
    --weight $weights_dir \
    --test-only

# Results:
# CaCO3 MSE: 17.116,     MSE of base model: 79.463,      R2: 0.785
# TOC   MSE: 0.045,     MSE of base model: 0.195,       R2: 0.772