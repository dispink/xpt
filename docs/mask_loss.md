# Do we calcualte the loss only on the masked patches?

[MAE](https://arxiv.org/abs/2111.06377) uses only the masked patches to calculate loss. We guess the intution is to avoid overestimation of performance because the model has the access to the unmasked patches and supposed to be able to predict them easily. However, this is not the case in the real world. Our optimal model on XRF spectra shows it has really hard time to predict the unmasked patches while the masked parts are predicted with high accuracy (Figure 1). Actually, MAE has the same problem in images. Figure 2 shows the predicted pixels of the unmasked patches are having bad quality. Therefore, we decided to include all patches in the loss calculation.

After include all patches as in the loss calculation, the overall reconstruction quality is improved significantly (Figure 3). The model training parameters are same as the one produce Figure 1.

![Figure 1](../files/spe_InstanceNorm()_optimal.png)
*Figure 1. Predicted and target spetra with masks, reconstructed by the model trained by the loss calculated only from the masked patches.*

![Figure 2](../files/fig2_in_MAE.png)

*Figure 2. Part of the example results from [MAE](https://arxiv.org/abs/2111.06377). For each triplet, it shows the masked image (left), recontructed image (middle), and the ground truth image (right).*

![Figure 3](../files/spe_InstanceNorm()_allpatches.png)
*Figure 1. Predicted and target spetra with masks, reconstructed by the model trained by the loss calculated from all patches.*