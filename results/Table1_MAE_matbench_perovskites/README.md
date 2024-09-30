# Table 1: Mean Absolute Errors for MatBench Task - Heat of Formation of Perovskites (matbench_perovskites)

This folder contains the data corresponding to **Table 1** from the paper, which presents the mean absolute errors (MAE) of various machine learning algorithms applied to the MatBench task of predicting the heat of formation for perovskites. Scripts to reproduce the results are available in the current folder as well.

### Algorithms Compared:
- **MEGNet (Matbench)** see the Matbench repository for details on [MEGNet implementations and results](https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_MegNet_kgcnn_v2.1.0/) for the Benchmark. The implementation in the benchmark is different from that used in this work! We used the default implementation on [MEGNet](https://github.com/materialsvirtuallab/megnet for all our tests.
- **MEGNet (this work, transferred embedding)**: Elemental embedding transferred from the formation energy task in the MEGNet repository, see [MEGNet Transferred Embedding Results](./MEGNet_transferred_embedding/). The model used followed the default configuration from original MEGNet repository.
- **MEGNet (this work, no embedding)**: The model was trained without using elemental embeddings, see [MEGNet No Embedding Results](./MEGNet_no_embedding/). The model used followed the default configuration from original MEGNet repository.
- **MODNet (Matbench)** 
- **MODNet (this work)** for the baseline model we used the DeBreuck2020 featurizer which is the default featurizer in [MODNet benchmarks in Matbench](https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.12/) and for the main MODNet publications. Our baseline result agrees with the Matbench benchmark and is available in the folder [MODNet_baseline](./MODNet_baseline/).
- **AutoMatMiner (Matbench)**  retrieved from the [MatBench repository](https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_automatminer_expressv2020/).
- **RF-SCM/MagPie (Matbench)** retrieved from the [MatBench repository](https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_rf/).

### Results:
| Algorithm                              | MAE (eV)                     |
|----------------------------------------|------------------------------|
| **MEGNet\***                           | 0.0352 (±0.0016)             |
| **MEGNet (this work, transferred embedding)** | 0.0685 (±0.0036)   |
| **MEGNet (this work, no embedding)**   | 0.0840 (±0.0058)             |
| **MODNet\***                           | 0.0908 (±0.0028)             |
| **MODNet (this work)**                 | 0.0888 (±0.0025)             |
| **AutoMatMiner\***                     | 0.2005 (±0.0085)             |
| **RF-SCM/MagPie\***                    | 0.2355 (±0.0034)             |

\* Data were retrieved from MatBench in May 2024.

### Citation:
If you use these results in your work, please cite the original paper:

- Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite dos Santos, *Boosting feature-based machine learning models for materials science: encoding electronic descriptors and graph-based features for enhanced accuracy and faster featurization*.
