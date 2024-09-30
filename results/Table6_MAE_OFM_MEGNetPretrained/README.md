# Table 6: Mean Absolute Errors for MODNet Models on Matbench Perovskites Task with encoded features and pretrained MEGNetPreL32 Features

This folder contains the data corresponding to **Table 6** from the paper, which presents the mean absolute errors (MAE) for MODNet models on the `matbench_perovskites` task. The table compares the inclusion of OFM latent features and both OFM latent features and MEGNetPreL32 features. In parentheses, the percentage MAE deviation from the default MatMiner featurizer in MODNet is shown.

### Features Compared:
- **Default MatMiner (MM)**: The default featurizer used in MODNet. Results in folder [Table1](../Table1_MAE_matbench_perovskites/MODNet_baseline).
- **MM + ℓ-OFM**: Combines the default MatMiner features with latent OFM features. This is the same as the `ℓ-OFM` model in the [Table2_MAE_MM_l-OFM folder](../Table2_MAE_MM_l-OFM/MM_OFM20cr/)
- **MM + MEGNetPreL32**: Combines the default MatMiner features with features from a pre-trained MEGNet model with 32 layers. This is the same model as the `MM + MEGNetPreL32` model in the [Table5](../Table5_MAE_MEGNetPretrainedModels) folder.
- **MM + ℓ-OFM + MEGNetPreL32**: Combines the default MatMiner features with latent OFM features and features from a pre-trained MEGNet model with 32 layers. Results available in the folder [./MM_l-OFM_MEGNetPreL32](./MM_l-OFM_MEGNetPreL32).
- **ℓ-MM**: Latent space reduction of MatMiner features. We used the compression ratio of 60%, which is provided in the [Table3_MAE_l-MM folder](../Table3_MAE_l-MM/l-MM_60cr_perovsk/).
- **ℓ-MM + ℓ-OFM**: Combines latent MatMiner features with latent OFM features. This is the same as the `ℓ-MM + ℓ-OFM` model in the [Table2_MAE_MM_l-OFM folder](../Table2_MAE_MM_l-OFM/MM_OFM20cr/).
- **ℓ-MM + MEGNetPreL32**: Combines latent MatMiner features with features from a pre-trained MEGNet model with 32 layers. Results available in the folder [./l-MM_MEGNetPreL32](./l-MM_MEGNetPreL32).
- **ℓ-MM + ℓ-OFM + MEGNetPreL32**: Combines latent MatMiner features with latent OFM features and features from a pre-trained MEGNet model with 32 layers. Results available in the folder [./l-MM_l-OFM_MEGNetPreL32](./l-MM_l-OFM_MEGNetPreL32).

### Results:
| Features                              | MAE (eV)                     |
|---------------------------------------|------------------------------|
| **Default MatMiner (MM)**             | 0.0888                       |
| **MM + ℓ-OFM**                        | 0.0743 (-16.2%)             |
| **MM + MEGNetPreL32**                 | 0.0726 (-18.2%)             |
| **MM + ℓ-OFM + MEGNetPreL32**         | 0.0629 (-29.1%)             |
| **ℓ-MM**                              | 0.0793 (-10.7%)             |
| **ℓ-MM + ℓ-OFM**                      | 0.0728 (-18.0%)             |
| **ℓ-MM + MEGNetPreL32**               | 0.0729 (-18.0%)             |
| **ℓ-MM + ℓ-OFM + MEGNetPreL32**       | 0.0653 (-26.5%)             |

### Citation:
If you use these results in your work, please cite the original paper:

- Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite dos Santos, *Boosting feature-based machine learning models for materials science: encoding electronic descriptors and graph-based features for enhanced accuracy and faster featurization*.
