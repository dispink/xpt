#! /bin/bash

mask_ratios=(0.7)
lrs=(1e-4)
scales=('instance_normalize')

for mask_ratio in ${mask_ratios[*]};
do
    for lr in ${lrs[*]};
    do
        for scale in ${scales[*]};
        do
            output_dir="results/HPtuning/pretrain-mask-ratio-$mask_ratio-blr-$lr-transform-$scale/"
            echo "START mask-ratio=$mask_ratio, blr=$lr, transform=$scale"
            python3 pretrain.py \
            --annotation_file data/pretrain/train/info.csv \
            --input_dir data/pretrain/train \
            --val_annotation_file data/pretrain/train/val.csv \
            --output_dir $output_dir \
            --verbose \
            --device cuda \
            --batch_size 256 \
            --epochs 100 \
            --mask_ratio $mask_ratio \
            --loss_mask_only \
            --blr $lr \
            --transform $scale
        done
    done
done
