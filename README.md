# MatterVial Support Data
This repository contains the data and code used to support the findings in the paper:

**"Combining feature-based approaches with graph neural networks and symbolic regression for synergistic performance and interpretability"**  
by Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite Santos

## Abstract
This study introduces MatterVial, an innovative hybrid framework for feature-based machine learning in materials science. MatterVial expands the feature space by integrating latent representations from a diverse suite of pretrained graph neural network (GNN) models—including structure-based (MEGNet), composition-based (ROOST), and equivariant (ORB) graph networks—with computationally efficient, GNN-approximated descriptors and novel features from symbolic regression. Our approach combines the chemical transparency of traditional feature-based models with the predictive power of deep learning architectures. When augmenting the feature-based model MODNet on Matbench tasks, this method yields significant error reductions and elevates its performance to be competitive with, and in several cases superior to, state-of-the-art end-to-end GNNs, with accuracy increases exceeding 40% for multiple tasks. An integrated interpretability module, employing surrogate models and symbolic regression, decodes the latent GNN-derived descriptors into explicit, physically meaningful formulas. This unified framework advances materials informatics by providing a high-performance, transparent tool that aligns with the principles of explainable AI, paving the way for more targeted and autonomous materials discovery.

## Repository Contents
This repository includes:
- **Results for the autoencoder and previous version of MatterVial (pGNN)**: The detailed results testing the different descriptor-oriented featurizers (l-MM and l-OFM)and all results obtained from the previously less general package which are now on the SI of the paper.
  - `results_encoders_and_SI_data/`
  
- **Featurized datasets and models**: Featurized datasets and final models for the main results of the paper. These are downloaded from [FigShare link](), a python script is provided to help with this.
 - download_datasets_and_models.py

- **Matbench v0.1 evaluation of MODNet@MatterVial models**: All general results
that are shown in Table 1 of the paper are shown
here including output files and fold by fold metrics.
  - `modnet_mattervial_matbench_results/`
Correspond to Table 1 of the paper:

<p align='center'>
<img src=".github/Table1MatterVial.png" alt="benchmark results">
</p>


Results table 2, focusing on matbench_perovskites dataset:

- **MODNet@MatterVial with adjacent models**:
  (COMING SOON)
  - `mattervial_perovskites_full_analysis/`
<p align='center'>
<img src=".github/Table2MatterVial.png" alt="benchmark results">
</p>

## Getting Started
To replicate the results presented in the paper or to apply these methods to your own dataset, check [MatterVial repository](https://github.com/rogeriog/MatterVial)
 enhanced with the GNN featurizers, and check the scripts provided along with the results.



## Contact and Contributions
For any clarifications, please contact the corresponding authors:  
- Rogério Almeida Gouvêa [rogeriog.em@gmail.com](mailto:rogeriog.em@gmail.com])  
- Gian-Marco Rignanese: [gian-marco.rignanese@uclouvain.be](mailto:gian-marco.rignanese@uclouvain.be)  
