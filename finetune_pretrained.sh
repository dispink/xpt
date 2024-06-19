#! /bin/bash

# pretraining parameters
mask_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
lrs=(1e-4 1e-5 1e-6)
scales=('instance_normalize' 'normalize' 'log')

# finetuning parameters
targets=(CaCO3 TOC)
epoch=100
warm_up=10
f_blr=1e-5

for mask_ratio in ${mask_ratios[*]};
do
    for lr in ${lrs[*]};
    do
        for scale in ${scales[*]};
        do
            for target in ${targets[*]};
            do
                output_dir="results/finetune_pretrained/pretrain-mask-ratio-$mask_ratio-blr-$lr-transform-$scale/$target" 
                echo "START $target mask-ratio=$mask_ratio, blr=$lr, transform=$scale" 
                python3 finetune.py \
                    --annotation_file data/finetune/${target}%/train/info.csv \
                    --input_dir data/finetune/${target}%/train \
                    --output_dir $output_dir \
                    --verbose \
                    --device cuda \
                    --pretrained_weight results/HPtuning/pretrain-mask-ratio-$mask_ratio-blr-$lr-transform-$scale/model.ckpt \
                    --batch_size 256 \
                    --epochs $epoch \
                    --warmup_epochs $warm_up \
                    --blr $f_blr \
                    --transform $scale \
                    --target_mean src/datas/xpt_${target}_target_mean.pth \
                    --target_std src/datas/xpt_${target}_target_std.pth
                
                python3 eval_finetune.py \
                    --target $target \
                    --annotation_file data/finetune/${target}%/train/info.csv \
                    --input_dir data/finetune/${target}%/train \
                    --output_dir $output_dir \
                    --transform $scale \
                    --weight "$output_dir/model.ckpt" 
            done
        done
    done
done