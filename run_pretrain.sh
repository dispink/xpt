#! /bin/bash

today=`date '+%Y%m%d'`;
output_dir=results/pretrain_test_$today
mkdir $output_dir

python pretrain.py \
    --annotation_file data/pretrain/train/info.csv \
    --input_dir data/pretrain/train\
    --output_dir $output_dir\
    --verbose \
    --device cuda \
    --mask_ratio 0.75 \
    --batch_size 256 \
    --epochs 100 \
    --blr 1e-5 \
  
