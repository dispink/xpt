# Useful links
1. [Discussion with Yu-Wen](https://docs.google.com/document/d/1IsWvWiVTuQ_j5wFc76Ls7NYi29niM4V6dsDlzO3wVyg/edit?usp=sharing)
1. [Exapmle](https://github.com/OPTML-Group/Unlearn-Sparse)
1. [Brief gothrough of folder structure and argparse](https://towardsdatascience.com/organizing-machine-learning-projects-e4f86f9fdd9c#:~:text=File%20structure,README.md%20file%20as%20well!)
1. [3-ways Generic Folder Structure for ML](https://dev.to/luxacademy/generic-folder-structure-for-your-machine-learning-projects-4coe)
1. [Python logging 中文教學](https://zhung.com.tw/article/python%E4%B8%AD%E7%9A%84log%E5%88%A9%E5%99%A8-%E4%BD%BF%E7%94%A8logging%E6%A8%A1%E7%B5%84%E4%BE%86%E6%95%B4%E7%90%86print%E8%A8%8A%E6%81%AF/)


# Command line arguments
I list the command line arguments that I think (1) should be included in the current stage and (2) might be included in the future when significantly scaling up. I don't include input_size as a varing parameter because it is fixed in the case. Different spectrum length (from different scanner series) will be padded to the same length, expecting a model capable in dealing with different scanner series.

## Pre-train
The arguments are partially adopted from [MAE_pretrain](https://github.com/facebookresearch/mae/blob/main/main_pretrain.py)

### Current stage
#### General parameters
- data_path
- batch_size
- epochs
- output_dir
- log_dir
- device: "cuda" or "cpu"
- seed
- num_workers: CPU workers for DataLoader
- pin_mem: Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.

#### Model parameters
- mask_ratio: The ratio of the masked patches

#### Optimizer parameters
- weight_decay (necessary? or simply adopt the optimal value from MAE?)
- blr: Base learning rate. I don't adopt the calculation of "absolute learning rate (`blr * total_batch_size / 256`)" in MAE because I don't see the need for it in our case.
- min_lr: The minimum learning rate of a cosine annealing schedule (necessary? or simply use the default value, 0?)
- warmup_epochs

### Future addition
- model: mae_vit_base_patch16/mae_vit_large_patch16
- accum_iter: Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
- resume: Resume training from a checkpoint
- distributed training parameters: world_size, local_rank:,dist_on_itp, etc

## Fine-tune
Here, I focus on the fine-tune process first. If it works out or needs to work with larger models, I might think about using PEFT, like Adapter and LoRA. The arguments are partially adopted from [MAE_finetune](https://github.com/facebookresearch/mae/blob/main/main_finetune.py).

### Current stage
#### General parameters
- data_path
- batch_size
- epochs
- output_dir
- log_dir
- device: "cuda" or "cpu"
- seed
- num_workers: CPU workers for DataLoader
- pin_mem: Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.

#### Optimizer parameters
- clip_grad: Clip gradient norm
- weight_decay (necessary? or simply adopt the optimal value from MAE?)
- blr: Base learning rate. I don't adopt the calculation of "absolute learning rate (`blr * total_batch_size / 256`)" in MAE because I don't see the need for it in our case.
- min_lr: The minimum learning rate of a cosine annealing schedule (necessary? or simply use the default value, 0?)
- layer_decay: Layer-wise lr decay from ELECTRA/BEiT
- warmup_epochs

#### Fine-tune parameters
- finetune: Finetune from which checkpoint
- global_pool: Use global pooling as the final layer for predictions
- cls_token: Use cls token for predictions

### Future addition
- model: base/large
- accum_iter: Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
- Other regularization params: augmentations, random erase, mixup, cutmix, etc.

## Evaluation
Evaluate models on the test set. The arguments come from my experience.

Pre-train: Spectrum reconstruction
Downstream tasks: Regression on TOC and CaCO3, respectively.

- data_path


# Planed structure
1. **.devcontainer**: Contain the configuration files for the development container, which is compatible to VScode Dev Container.

1. **data**: Contains all the data used in the project. It is further divided into subfolders:
    - _raw_: Raw spectra in the Avaatech XRF Core Scanner format. Each subfolder contains the raw data for a core series.
    - previously compiled data
    - pre-training, 
    - fine-tuning
    - spe?
    - info?

1. **notebooks**: Collect Jupyter notebooks for experimentation, analysis, and model development.

1. **configs**: Store configuration files or parameters used in your project, such as hyperparameters, model configurations, or experiment settings.

1. **docs**: Include any project-related documentation, such as README files, data dictionaries, or project specifications.

1. **logs**: Store log files generated during model training, evaluation, or other experiments.

1. **models**: Store all the trained models. It is further divided into subfolders:
    - _pre-trained_: Pre-trained models.
    - _fine-tuned_: Fine-tuned models.
    - (_inference_: Models used for inference and prediction.) Maybe in the future.

1. **src**: Contain all the scripts used in the project. It is further divided into subfolders:
    - _data_: Scripts for data preprocessing, and data loading.
    - _models_: Scripts for model architectures, loss functions, and evaluation metrics.
    - _training_: Scripts for training and evaluation.
    - (_inference_: Scripts for inference and prediction.) Maybe in the future.

# Rename
Previously, I simply adopt MAE scripts for modification. Now, I think it's better to rename key model names to be more specific to our project.

- Masked Autoencoder: mae -> max (Masked Autoencoder for XRF spectra). Meanwhile, I remove the components of vit and patch16 from the pre-trained model name to shorten the name. e.g., max_base instead of mae_vit_base_patch16.
- Encoder: vit -> vitx. MAE simply uses the Vit architecutre, which is identical to the MAE model removing decoder part. But in our case, the input data size is changed from 224X224 to 1X2048, which consequently changes the patchfication. The Vit architecture is no longer identical to our model, but with some modification. Therefore, I think it's better to rename it to vitx (ViT for XRF spectra) to avoid confusion.
- Project name?