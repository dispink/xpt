#! /bin/bash

mask_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
lrs=(1e-4 1e-5 1e-6)
scales=('instance_normalize' 'normalize' 'log')

for mask_ratio in ${mask_ratios[*]};
do
    for lr in ${lrs[*]};
    do
        for scale in ${scales[*]};
        do
            output_dir="results/pretrain-mask-ratio-$mask_ratio-blr-$lr-transform-$scale/"
            echo "START mask-ratio=$mask_ratio, blr=$lr, transform=$scale"
            python3 pretrain.py \
            --annotation_file data/pretrain/train/info.csv \
            --input_dir data/pretrain/train \
            --output_dir $output_dir \
            --verbose \
            --device cuda \
            --batch_size 256 \
            --epochs 0 \
            --mask_ratio $mask_ratio \
            --blr $lr \
            --transform $scale
        done
    done
done