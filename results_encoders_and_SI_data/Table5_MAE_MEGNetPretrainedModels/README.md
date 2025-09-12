# Table 5: Mean Absolute Errors for MODNet Models on Matbench Perovskites Task and Subsets

This folder contains the data corresponding to **Table 5** from the paper, which presents the mean absolute errors (MAE) for MODNet models on the `matbench_perovskites` task and its subsets. The table compares the inclusion of features from pre-trained MEGNet models distributed by Materials Virtual Lab. N represents the size of the dataset used for the prediction. In parentheses, the percentage MAE deviation from the default MatMiner featurizer in MODNet for each task is shown.

### Features Compared:
- **Default MatMiner (MM)**: The default featurizer used in MODNet, which includes 1020 features. Besides the baseline model with the default MatMiner features in the entire set (present in folder [Table1](../Table1_MAE_matbench_perovskites/MODNet_baseline), we also compare the performance of the model with the default MatMiner features in subsets of the dataset with 5,000 ([./MM/subset5000](./MM/subset5000))and 1,000 samples ([./MM/subset1000](./MM/subset1000)).
- **MM + MEGNetPreL16**: Combines the default MatMiner features with features from a pre-trained MEGNet model with 16 layers. Results for all dataset sizes are available in the folder [./MM_MEGNetPreL16](./MM_MEGNetPreL16).
- **MM + MEGNetPreL32**: Combines the default MatMiner features with features from a pre-trained MEGNet model with 32 layers. Results for all dataset sizes are available in the folder [./MM_MEGNetPreL32](./MM_MEGNetPreL32).

### Results:
| Features                              | MAE (eV) matbench perovskites (N=18,928) | MAE (eV) matbench perovskites (N=5,000) | MAE (eV) matbench perovskites (N=1,000) |
|---------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|
| **Default MatMiner (MM)**             | 0.0888                                   | 0.1667                                   | 0.2802                                   |
| **MM + MEGNetPreL16**                 | 0.0752 (-15.3%)                         | 0.1202 (-27.9%)                         | 0.1862 (-33.5%)                         |
| **MM + MEGNetPreL32**                 | 0.0726 (-18.2%)                         | 0.1167 (-30.0%)                         | 0.1749 (-37.6%)                         |

### Citation:
If you use these results in your work, please cite the original paper:

- Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite dos Santos, *Boosting feature-based machine learning models for materials science: encoding electronic descriptors and graph-based features for enhanced accuracy and faster featurization*.

