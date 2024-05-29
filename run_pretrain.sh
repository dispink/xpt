python pretrained.py \
    --annotation_file data/pretrain/train/info.csv \
    --input_dir data/pretrain/train/spe/ \
    --output_dir results/ \
    --verbose \
    --device cuda \
    --mask_ratio 0.75 \
    --batch_size 256 \
    --epochs 100 \
    --blr 1e-5 \
    --annealing_epochs 90
  
