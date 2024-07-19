#! /bin/bash

target=(CaCO3 TOC)

annotation_files=(
    'info_10.csv'
    'info_50.csv'
    'info_100.csv'
    'info_500.csv'
    'info_1000.csv'
    'info.csv'
)

pretrain_blr="1e-4"
mask_ratio="0.5"
scale="instance_normalize"
pretrained_weight="results/HPtuning-loss-on-masks/pretrain-mask-ratio-${mask_ratio}-blr-${pretrain_blr}-transform-${scale}/model.ckpt"

for target in ${target[*]};
do
    input_dir=data/finetune/${target}%/train

    if [ "$target" == "CaCO3" ]; then
    epochs=(100 150 150 200 200 150)
    warm_ups=(10 15 15 20 20 15)
    lrs=(1e-4 1e-4 1e-4 1e-5 1e-5 1e-5)

    elif [ "$target" == "TOC" ]; then
        epochs=(50 50 50 75 100 100)
        warm_ups=(5 5 5 8 10 10)
        lrs=(1e-4 1e-5 1e-4 1e-4 1e-4 1e-4)
    fi

    for (( i=0; i<${#annotation_files[*]}; ++i));
    do
        epoch=${epochs[$i]}
        warm_up=${warm_ups[$i]}
        annotation_file=${annotation_files[$i]}
        lr=${lrs[$i]}
        output_dir=results/finetune_data_amount/${target}-${annotation_file}-epochs-${epoch}-blr-${lr}/
        echo "START $target train=$annotation_file, blr=$lr, epochs=$epoch, warm_up=$warm_up"
        python finetune.py \
            --annotation_file ${input_dir}/${annotation_file} \
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
