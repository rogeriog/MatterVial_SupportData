# Table 4: Mean Absolute Errors for MODNet Models on Matbench Perovskites Task considering latent MM and OFM Features with and without MEGNet Proxy

This folder contains the data corresponding to **Table 4** from the paper, which presents the mean absolute errors (MAE) for MODNet models on the `matbench_perovskites` task. The table compares the inclusion of latent features originally obtained from the autoencoder and through the MEGNet featurizers. In parentheses, the percentage MAE deviation from the default MatMiner featurizer in MODNet is shown.

### Features Compared:
- **Default MatMiner (MM)**: The default featurizer used in MODNet, which includes 1020 features. This is the baseline for comparison. Data available in the [Table1_MAE_matbench_perovskites folder](../Table1_MAE_matbench_perovskites/MODNet_baseline).
- **ℓ-MM**: Latent space reduction of MatMiner features. We used the compression ratio of 60%, which is provided in the [Table3_MAE_l-MM folder](../Table3_MAE_l-MM/l-MM_60cr_perovsk/).
- **MEGNet ℓ-MM**: Latent space reduction of MatMiner features using MEGNet model as proxy. Results available in folder [Table4_MAE_MEGNetProxy_MM_OFM/pl-MM/](./pl-MM).
- **MM + ℓ-OFM**: Combines the default MatMiner features with latent OFM features. This is the same as the `ℓ-OFM` model in the [Table2_MAE_MM_l-OFM folder](../Table2_MAE_MM_l-OFM/MM_OFM20cr/).
- **MM + MEGNet ℓ-OFM**: Combines the default MatMiner features with latent OFM features obtained using MEGNet as proxy. Results available in folder [Table4_MAE_MEGNetProxy_MM_OFM/MM_pl-OFM/](./MM_pl-OFM).
- **MEGNet ℓ-MM + MEGNet ℓ-OFM**: Combines latent MatMiner features and latent OFM features, both obtained using MEGNet models as proxy. Results available in folder [Table4_MAE_MEGNetProxy_MM_OFM/pl-MM_pl-OFM/](./pl-MM_pl-OFM).

### Results:
| Features                              | MAE (eV)                     |
|---------------------------------------|------------------------------|
| **Default MatMiner (MM)**             | 0.0888                       |
| **ℓ-MM**                              | 0.0793 (-10.7%)             |
| **MEGNet ℓ-MM**                       | 0.1052 (+18.5%)             |
| **MM + ℓ-OFM**                        | 0.0743 (-16.2%)             |
| **MM + MEGNet ℓ-OFM**                 | 0.0794 (-10.6%)             |
| **MEGNet ℓ-MM + MEGNet ℓ-OFM**        | 0.0973 (+9.6%)              |

### Citation:
If you use these results in your work, please cite the original paper:

- Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite dos Santos, *Boosting feature-based machine learning models for materials science: encoding electronic descriptors and graph-based features for enhanced accuracy and faster featurization*.
