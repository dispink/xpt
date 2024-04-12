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
