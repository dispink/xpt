#! /bin/bash

target=(CaCO3 TOC)

annotation_files=(
    'info_train_10.csv'
    'info_train_50.csv'
    'info_train_100.csv'
    'info_train_500.csv'
    'info_train_1000.csv'
    'info_train_1500.csv'
    'info.csv'
)

scale='instance_normalize'

for target in ${target[*]};
do
    input_dir=data/finetune/${target}%/train

    if [ "$target" == "CaCO3" ]; then
    epochs=(10 100 75 100 100 100 100)
    warm_ups=(1 10 8 10 10 10 10)
    lrs=(1e-5 1e-5 1e-5 1e-5 1e-6 1e-5 1e-5)

    elif [ "$target" == "TOC" ]; then
        epochs=(25 100 100 100 100 100 100)
        warm_ups=(3 10 10 10 10 10 10)
        lrs=(1e-5 1e-6 1e-5 1e-5 1e-5 1e-4 1e-5)
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
            --annotation_file ${input_dir}/${annotation_files[$i]} \
            --input_dir $input_dir \
            --output_dir $output_dir \
            --verbose \
            --device cuda \
            --pretrained_weight results/HPtuning/pretrain-mask-ratio-0.9-blr-1e-4-transform-${scale}/model.ckpt \
            --batch_size 256 \
            --epochs $epoch \
            --warmup_epochs $warm_up \
            --blr $lr \
            --transform $scale \
            --target_mean src/datas/xpt_${target}_target_mean.pth \
            --target_std src/datas/xpt_${target}_target_std.pth
        python eval_finetune.py \
            --target $target \
            --annotation_file ${input_dir}/info.csv \
            --input_dir $input_dir \
            --output_dir $output_dir \
            --transform $scale \
            --weight "$output_dir/model.ckpt" 
    done
done
