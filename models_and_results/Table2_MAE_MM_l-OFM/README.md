# Table 2: Mean Absolute Errors for MODNet Models on MatBench Perovskites Task

This folder contains the data corresponding to **Table 2** from the paper, which presents the mean absolute errors (MAE) of various MODNet models applied to the MatBench task of predicting the heat of formation for perovskites. These models include pristine OFM features and different latent space reductions of OFM features in addition to the default MatMiner features. Scripts to reproduce the results are available in the current folder as well.

### Features Compared:
- **Default MatMiner (MM)**: The default featurizer used in MODNet, which includes 1020 features. Same as [Table 1](../Table1_MAE_matbench_perovskites/MODNet_baseline) data.
- **MM + original OFM**: Combines the default MatMiner features with the original OFM features, resulting in 1020 + 943 features. Results present in folder [MM_OFM](./MM_OFM).
- **MM + latent OFM 20% c.r. (ℓ-OFM)**: Combines the default MatMiner features with a latent space reduction of OFM features using a 20% compression ratio, resulting in 1020 + 188 features. Results present in folder [MM_OFM20cr](./MM_OFM20cr).
- **MM + latent OFM 10% c.r.**: Combines the default MatMiner features with a latent space reduction of OFM features using a 10% compression ratio, resulting in 1020 + 94 features. Results present in folder [MM_OFM10cr](./MM_OFM10cr).
- **MM + PCA reduced OFM (n=188)**: Combines the default MatMiner features with a PCA-reduced representation of OFM features with the same dimensions as the 20% compression ratio, resulting in 1020 + 188 features. Results present in folder [MM_OFMPCA](./MM_OFMPCA).

### Results:
| Features                              | n                            | MAE (eV)                     |
|---------------------------------------|------------------------------|------------------------------|
| **Default MatMiner (MM)**             | 1020                         | 0.0888                       |
| **MM + original OFM**                 | 1020 + 943                   | 0.0751 (-15.3%)             |
| **MM + latent OFM 20% c.r. (ℓ-OFM)**  | 1020 + 188                   | 0.0743 (-16.2%)             |
| **MM + latent OFM 10% c.r.**           | 1020 + 94                    | 0.0777 (-12.4%)             |
| **MM + PCA reduced OFM (n=188)**       | 1020 + 188                   | 0.0748 (-15.7%)             |

### Citation:
If you use these results in your work, please cite the original paper:

- Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite dos Santos, *Boosting feature-based machine learning models for materials science: encoding electronic descriptors and graph-based features for enhanced accuracy and faster featurization*.
