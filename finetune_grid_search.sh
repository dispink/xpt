#! /bin/bash

target=CaCO3
annotation_files=(
    'info_train_10.csv'
    'info_train_50.csv'
    'info_train_100.csv'
    'info_train_500.csv'
    'info_train_1000.csv'
    'info_train_1500.csv'
    'info.csv'
)

epochs=(10 25 50 75 100)
warm_ups=(1 3 5 8 10)
lrs=(1e-4 1e-5 1e-6)


for annotation_file in ${annotation_files[*]};
do
    for lr in ${lrs[*]};
    do
        for (( i=0; i<${#epochs[*]}; ++i));
        do
            epoch=${epochs[$i]}
            warm_up=${warm_ups[$i]}
            output_dir="results/finetune-$target-$annotation_file-epochs-$epoch-blr-$lr/"
            echo "START $target train=$annotation_file, blr=$lr, epochs=$epoch, warm_up=$warm_up"
            python finetune.py \
            --annotation_file data/finetune/${target}%/train/$annotation_file \
            --input_dir data/finetune/${target}%/train \
            --val_annotation_file data/finetune/${target}%/train/val.csv \
            --val_input_dir data/finetune/${target}%/train \
            --output_dir $output_dir \
            --verbose \
            --device cuda \
            --pretrained_weight results/HPtuning/pretrain-mask-ratio-0.5-blr-1e-4-transform-instance_normalize/model.ckpt \
            --batch_size 256 \
            --epochs $epoch \
            --warmup_epochs $warm_up \
            --blr $lr \
            --target_mean src/datas/xpt_${target}_target_mean.pth \
            --target_std src/datas/xpt_${target}_target_std.pth
            python eval_finetune.py \
            --annotation_file data/finetune/${target}%/train/val.csv \
            --input_dir data/finetune/${target}%/train \
            --test-only \
            --transform instance_normalize \
            --target $target \
            --weight "$output_dir/model.ckpt"
        done
    done
done