# XRF spectrum pre-training

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
    - **data**: Scripts for data preprocessing, and data loading.
    - **models**: Scripts for model architectures, loss functions, and evaluation metrics.
    - **training**: Scripts for training and evaluation.
    - **inference**: Scripts for inference and prediction on the test or new data.
    - **utils**: Utility scripts for logging and other helper functions.

1. **archives**: Store old or deprecated scripts, models, or data. 

1. **pilot**: Store pilot experiments before integrating into the main project.