#! /bin/bash

# Evaluate the optimal finetuned model on the test set

target=TOC
input_dir=data/finetune/${target}%/test
pretrain_blr="1e-4"
mask_ratio="0.5"
scale=instance_normalize
weights_dir=results/finetune_pretrained/pretrain-mask-ratio-${mask_ratio}-blr-${pretrain_blr}-transform-${scale}/${target}/model.ckpt
#weights_dir=results/checkpoint_from_scratch/${target}_from_scratch.ckpt

python eval_finetune.py \
    --target $target \
    --annotation_file ${input_dir}/info.csv \
    --input_dir $input_dir \
    --transform $scale \
    --weight $weights_dir \
    --test-only

# Results:
# Optimal models
# CaCO3 MSE: 11.413,    MSE of base model: 455.999,     R2: 0.975
# TOC   MSE: 0.095,     MSE of base model: 0.097,       R2: 0.018