#! /bin/bash

target=(CaCO3 TOC)

annotation_files=(
    'info_10.csv'
    'info_50.csv'
    'info_100.csv'
    'info_150.csv'
    'info_200.csv'
    'info_250.csv'
    'info_train.csv'
)

pretrain_blr="1e-4"
mask_ratio="0.5"
scale="instance_normalize"

epoch=100
warm_up=10
lr=1e-4

pretrained_weight="results/HPtuning-loss-on-masks/pretrain-mask-ratio-${mask_ratio}-blr-${pretrain_blr}-transform-${scale}/model.ckpt"

for target in ${target[*]};
do
    input_dir=data/finetune/${target}%/test
    for (( i=0; i<${#annotation_files[*]}; ++i));
    do
        output_dir=results/finetune_data_amount_case/${target}-${annotation_files[i]}-epochs-${epoch}-blr-${lr}/
        echo "START $target train=$annotation_file, blr=$lr, epochs=$epoch, warm_up=$warm_up"
        python finetune.py \
            --annotation_file ${input_dir}/${annotation_files[i]} \
            --val_annotation_file ${input_dir}/val.csv \
            --input_dir $input_dir \
            --val_input_dir $input_dir\
            --output_dir $output_dir \
            --verbose \
            --device cuda \
            --pretrained_weight $pretrained_weight \
            --batch_size 256 \
            --epochs $epoch \
            --warmup_epochs $warm_up \
            --blr $lr \
            --transform $scale \
            --target_mean src/datas/xpt_${target}_target_mean.pth \
            --target_std src/datas/xpt_${target}_target_std.pth
        python eval_finetune.py \
            --target $target \
            --annotation_file ${input_dir}/val.csv \
            --input_dir $input_dir \
            --output_dir $output_dir \
            --transform $scale \
            --weight "$output_dir/model.ckpt" \
            --test-only
    done
done
