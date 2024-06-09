today=`date '+%Y%m%d'`;
output_dir=results/finetune_test_$today
mkdir $output_dir

python finetune.py \
    --annotation_file data/finetune/TOC%/train/info.csv \
    --input_dir data/finetune/TOC%/train \
    --output_dir $output_dir \
    --verbose \
    --device cuda \
    --pretrained_weight models/mae_vit_base_patch16_l-coslr_1e-05_20231229.pth \
    --batch_size 256 \
    --epochs 100 \
    --blr 1e-6 \