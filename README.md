# Pretraining Foundation Models: <br> Unleashing the Power of Forgotten Spectra for Advanced Geological Applications
**X-ray fluorescence (XRF)** core scanning is renowned for its highresolution, non-destructive, and user-friendly operation. Despite the
extensive applications of XRF data, the universal quantification of this data into specific geological proxies remains challenging due to
the inherent __non-linearity__ and __project-scale limitation__.

Our study aims to address the challenges by harnessing two interdisciplinary advancements: 
1. Vast amount of XRF spectra acquired from series of scientific drilling programs
1. More powerful training scheme and complex model inspired by the success of large language models (LLMs).

We proposed a pretraining-finetuning framework that leverages the vast amount of XRF spectra to pretrain a foundation model. **Masked Spectrum Modeling (MSM)** is modifed from [BERT](https://arxiv.org/abs/1810.04805), [ViT](https://arxiv.org/abs/2010.11929), and [MAE](https://arxiv.org/abs/2111.06377) to our pretraining process. It is designed to let our foundation model learn the underlying patterns and relationships in the XRF spectra, which can be transferred to downstream tasks. The pretraining process is followed by fine-tuning the model on specific geological proxies to adapt the model to the target tasks. Hence, the downstream fine-tuning does not necessary require large amount of labeled data, which is contrast to the conventional method training a model from scratch in each project. 

<p align="center">
  <img src="results/demo_patch_combined.png" width="360" />
</p>

### The work has been published on EGU2024 as a poster:
Lee, A.-S., Lin, H.-T., and Liou, S. Y. H.: Pretraining Foundation Models: Unleashing the Power of Forgotten Spectra for Advanced Geological Applications, EGU General Assembly 2024, Vienna, Austria, 14–19 Apr 2024, EGU24-4956, https://doi.org/10.5194/egusphere-egu24-4956, 2024.

# Environment setup
## Docker container
We adopt the container template, `cuda118`, from <https://github.com/dispink/docker-example>.

### Versions of the main packages
-   Python 3.11
-   CUDA 11.8
-   cudnn 8.6.0

## Folder structure
1. **.devcontainer**: Contain the configuration files for the Docker container, which is compatible to VScode Dev Container.

1. **data**: Contain all the data used in the project. It is further divided into subfolders:
    - **raw**: Raw spectra in the Avaatech XRF Core Scanner format. Each subfolder contains the raw data for a core series.
    - **legacy**: Previously compiled and raw data [(Lee et al., 2022)](https://doi.org/10.1038/s41598-022-25377-x). 
    - **pretrain**: Data used for pre-training and is built from the previously compiled spectra data `spe_dataset_20220629.csv`. It is further divided into subfolder sutructure as below.

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
    There is no validation set in folders because it will be randomly sampled during training. The test set is composed of three cores ('PS75-056-1', 'LV28-44-3', 'SO264-69-2') isolated from the beginning and not used in the pre-training and fine-tuning process. I should test the model only at the very last step of the project, otherwise may introduce data leakage and over-estimate the model's generalization ability. The script is `src/datas/build_data.py`.

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
    - **inference**: Scripts for inference and prediction on the test or new data.
    - **utils**: Utility scripts for logging and other helper functions.

1. **archives**: Store old or deprecated scripts, models, or data. 

1. **pilot**: Store pilot experiments before integrating into the main project.