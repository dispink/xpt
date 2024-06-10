target=CaCO3
today=`date '+%Y%m%d'`;
output_dir=results/finetune_test_${target}_$today
mkdir $output_dir

python finetune.py \
    --annotation_file data/finetune/${target}%/train/info.csv \
    --input_dir data/finetune/${target}%/train \
    --output_dir $output_dir \
    --verbose \
    --device cuda \
    --pretrained_weight results/pretrain_test_20240609/model.ckpt \
    --batch_size 256 \
    --epochs 100 \
    --blr 1e-6 \
    --target_mean src/datas/xpt_${target}_target_mean.pth \
    --target_std src/datas/xpt_${target}_target_std.pth