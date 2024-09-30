# Table 3: Evaluation of the effects of dimensionality reduction on default MatMiner features used on MODNet model on Matbench tasks matbench_perovskites and matbench_mp_gap

This folder contains the data corresponding to **Table 3** from the paper, which presents the evaluation of the effects of dimensionality reduction on default MatMiner features used on the MODNet model for the Matbench tasks `matbench_perovskites` and `matbench_mp_gap`. The table includes the number of features (n) and the mean absolute errors (MAE) for each model. In parentheses, the percentage MAE deviation from the default MatMiner featurizer in MODNet for each task is shown.

### Features Compared:
- **Default MatMiner**: The default featurizer used in MODNet, which includes 1020 features for `matbench_perovskites` and 1264 features for `matbench_mp_gap`.
- **Latent MatMiner without compression (1:1 latent space)**: The MatMiner features without compression, resulting in 1264 features for both tasks. Results present in folder [l-MM_no_compression_perovsk](./l-MM_no_compression_perovsk) and [l-MM_no_compression_mp_gap](./l-MM_no_compression_mp_gap).
- **Latent MatMiner 80% c.r.**: Latent space reduction of MatMiner features using an 80% compression ratio, resulting in 1011 features for both tasks. Results present in folder [l-MM_80cr_perovsk](./l-MM_80cr_perovsk) and [l-MM_80cr_mp_gap](./l-MM_80cr_mp_gap).
- **Latent MatMiner 60% c.r. (ℓ-MM)**: Latent space reduction of MatMiner features using a 60% compression ratio, resulting in 758 features for both tasks. Results present in folder [l-MM_60cr_perovsk](./l-MM_60cr_perovsk) and [l-MM_60cr_mp_gap](./l-MM_60cr_mp_gap).
- **Latent MatMiner 40% c.r.**: Latent space reduction of MatMiner features using a 40% compression ratio, resulting in 505 features for both tasks. Results present in folder [l-MM_40cr_perovsk](./l-MM_40cr_perovsk) and [l-MM_40cr_mp_gap](./l-MM_40cr_mp_gap).
- **PCA reduced MatMiner (n=758)**: PCA-reduced representation of MatMiner features with the same dimensions as the 60% compression ratio, resulting in 758 features for both tasks. Results present in folder [PCA_MM_perovsk](./PCA_MM_perovsk) and [PCA_MM_mp_gap](./PCA_MM_mp_gap).

### Results:
| Features used                                      | Task                  | n    | MAE (eV)       |
|----------------------------------------------------|-----------------------|------|----------------|
| **Default MatMiner**                               | matbench perovskites  | 1020 | 0.0888 ±0.0028 |
| **Latent MatMiner without compression (1:1 latent space)** | matbench perovskites  | 1264 | 0.0767 (-13.6%) |
| **Latent MatMiner 80% c.r.**                       | matbench perovskites  | 1011 | 0.0788 (-11.3%) |
| **Latent MatMiner 60% c.r. (ℓ-MM)**                | matbench perovskites  | 758  | 0.0793 (-10.7%) |
| **Latent MatMiner 40% c.r.**                       | matbench perovskites  | 505  | 0.0844 (-4.9%) |
| **PCA reduced MatMiner (n=758)**                   | matbench perovskites  | 758  | 0.0816 (-8.1%) |
| **Default MatMiner**                               | matbench mp_gap       | 1264 | 0.2724 ±0.0052 |
| **Latent MatMiner without compression (1:1 latent space)** | matbench mp_gap       | 1264 | 0.2542 (-6.7%) |
| **Latent MatMiner 80% c.r.**                       | matbench mp_gap       | 1011 | 0.2809 (+3.1%) |
| **Latent MatMiner 60% c.r. (ℓ-MM)**                | matbench mp_gap       | 758  | 0.2911 (+6.8%) |
| **Latent MatMiner 40% c.r.**                       | matbench mp_gap       | 505  | 0.3280 (+20.4%) |
| **PCA reduced MatMiner (n=758)**                   | matbench mp_gap       | 758  | 0.2968 (+8.9%) |

### Citation:
If you use these results in your work, please cite the original paper:

- Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite dos Santos, *Boosting feature-based machine learning models for materials science: encoding electronic descriptors and graph-based features for enhanced accuracy and faster featurization*.