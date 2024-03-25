# Restructure the project: Command line logic and folder structure

## Useful links
1. [Discussion with Yu-Wen](https://docs.google.com/document/d/1IsWvWiVTuQ_j5wFc76Ls7NYi29niM4V6dsDlzO3wVyg/edit?usp=sharing)
1. [Exapmle](https://github.com/OPTML-Group/Unlearn-Sparse)
1. [Brief gothrough of folder structure and argparse](https://towardsdatascience.com/organizing-machine-learning-projects-e4f86f9fdd9c#:~:text=File%20structure,README.md%20file%20as%20well!)
1. [3-ways Generic Folder Structure for ML](https://dev.to/luxacademy/generic-folder-structure-for-your-machine-learning-projects-4coe)
1. [Python logging 中文教學](https://zhung.com.tw/article/python%E4%B8%AD%E7%9A%84log%E5%88%A9%E5%99%A8-%E4%BD%BF%E7%94%A8logging%E6%A8%A1%E7%B5%84%E4%BE%86%E6%95%B4%E7%90%86print%E8%A8%8A%E6%81%AF/)

## Command line arguments
I list the command line arguments that I think (1) should be included in the current stage and (2) might be included in the future when significantly scaling up. I don't include input_size as a varing parameter because it is fixed in the case. Different spectrum length from different scanner series will be padded to the same length, expecting a model capable in dealing with different scanner series.

### Pre-train
The arguments are partially adopted from [MAE_pretrain](https://github.com/facebookresearch/mae/blob/main/main_pretrain.py).

#### General parameters
- data_path
- batch_size
- epochs
- output_dir
- log_dir
- device: "cuda" or "cpu"
- num_workers: CPU workers for DataLoader
- pin_mem: Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.

#### Model parameters
- mask_ratio: The ratio of the masked patches

#### Optimizer parameters
- weight_decay (necessary? or simply adopt the optimal value from MAE?)
- blr: Base learning rate. I don't adopt the calculation of "absolute learning rate (`blr * total_batch_size / 256`)" in MAE because I don't see the need for it in our case.
- min_lr: The minimum learning rate of a cosine annealing schedule (necessary? or simply use the default value, 0?)
- warmup_epochs

#### Future addition
- model: base/large/huge
- accum_iter: Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
- resume: Resume training from a checkpoint
- distributed training parameters: world_size, local_rank,dist_on_itp, etc

### Fine-tune
Here, I focus on the end-to-end fine-tuning process, first. If it works or needs to work with larger models, I might think about using PEFT, like Adapter and LoRA. The arguments are partially adopted from [MAE_finetune](https://github.com/facebookresearch/mae/blob/main/main_finetune.py).

#### General parameters
- data_path
- batch_size
- epochs
- output_dir
- log_dir
- device: "cuda" or "cpu"
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
- finetune: Finetune from which pre-trained checkpoint
- global_pool: Use global pooling to include all features except cls token for predictions
- cls_token: Useonly  cls token for predictions

#### Future addition
- model: base/large/huge
- accum_iter: Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
- Other regularization params: augmentations, random erase, mixup, cutmix, etc.

### Evaluation
Evaluate models on the test set. The arguments come from my experience and goals. The test set is three cores ('PS75-056-1', 'LV28-44-3', 'SO264-69-2') isolated from the beginning and not used in the pre-training and fine-tuning process.

Pre-train task: Spectrum reconstruction<br>
Downstream tasks: Regression on TOC and CaCO3, respectively.

- pretrain: test on the pretrain task
- downstream: test on downstream tasks
- data_path
- checkpoint: The checkpoint to be evaluated
- batch_size
- output_dir
- log_dir
- device

#### Future addition
- model: base/large/huge

## Folder structure
1. **.devcontainer**: Contain the configuration files for the Docker container, which is compatible to VScode Dev Container.

1. **data**: Contain all the data used in the project. It is further divided into subfolders:
    - **raw**: Raw spectra in the Avaatech XRF Core Scanner format. Each subfolder contains the raw data for a core series.
    - **legacy**: Previously compiled data [(Lee et al., 2022)](https://doi.org/10.1038/s41598-022-25377-x).
    - **pretrain**: Data used for pre-training. It is further divided into subfolder sutructure as below.

    ```    
        +- train
            +- spe
            +- info.csv
        +- test
            (same as in train)
    ```
    - **fine-tune**: Data used for fine-tuning. It is further divided into subfolder sutructure as below
    ``` 
        +- CaCO3   
            +- train
                +- spe
                +- target
                +- info.csv
            +- test
                (same as in train)
        +- TOC
            (same as in CaCO3)
    ```
    There is no validation set in folders because it will be randomly sampled during training. The test set is three cores ('PS75-056-1', 'LV28-44-3', 'SO264-69-2') isolated from the beginning and not used in the pre-training and fine-tuning process. I should test the model only at the very last step of the project, otherwise may introduce data leakage and over-estimate the model's generalization ability.

1. **notebooks**: Collect Jupyter notebooks for experimentation, analysis, and model development.

1. **configs**: Store configuration files or parameters used in the project, such as hyperparameters, model configurations, or experiment settings.

1. **docs**: Include any project-related documentation, such as data dictionaries, or project specifications.

1. **results**: Store output files, reports, or visualizations.

1. **logs**: Store log files generated during model training, evaluation, or other experiments.

1. **models**: Store all the trained models. It is further divided into subfolders:
    - **pre-trained**: Pre-trained models.
    - **fine-tuned**: Fine-tuned models.

1. **src**: Contain all the scripts used in the project. It is further divided into subfolders:
    - **datas**: Scripts for data preprocessing, and data loading.
    - **models**: Scripts for model architectures, loss functions, and evaluation metrics.
    - **train**: Scripts for training and related functions.
    - **eval**: Scripts for evalutaion and related functions.
    - **training**: Scripts for training and evaluation.
    - **inference**: Scripts for inference and prediction on the test or new data.
    - **utils**: Utility scripts for logging and other helper functions.

1. **archive**: Store old or deprecated scripts, models, or data. 

1. **pilot**: Store pilot experiments before integrating into the main project.

## Model renaming
Previously, I simply adopted MAE scripts for modification. Now, I think it's better to rename key models to be more specific to our project.

- Masked Autoencoder: mae -> max (Masked Autoencoder for XRF spectra). Meanwhile, I remove the components of vit and patch16 from the pre-trained model name to shorten the name because this info is not critical. e.g., max_base instead of mae_vit_base_patch16.
- Encoder: vit -> vitx. MAE simply uses the Vit architecutre, which is identical to the MAE model removing decoder part. But in our case, the input data size is changed from 224X224 to 1X2048, which consequently changes the patchfication. The Vit architecture is no longer identical to our encoder. Therefore, I think it's better to rename it to vitx (ViT for XRF spectra) to avoid confusion.
- Project name from xpt to??