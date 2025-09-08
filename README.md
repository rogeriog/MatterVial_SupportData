# Supporting Data: Bridging Feature-Based Models and Graph Neural Networks

This repository contains the data and code used to support the findings in the paper:

**"Bridging Feature-Based Models and Graph Neural Networks: A Hybrid Approach for Accurate and Interpretable Materials Modeling"**  
by Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite Santos

## Abstract
This study presents a method to improve feature-based machine learning models in materials science by leveraging graph neural networks (GNNs) as featurizers. Our approach combines traditional chemical and geometrical features with GNN-derived descriptors, offering both interpretability and enhanced accuracy. Three independent techniques for GNN featurizer construction were employed:  
- Compressing electronic structure descriptors  
- Integrating pretrained GNN models  
- Task-specific GNN development

Using MODNet as the base feature-based model and MEGNet for GNN featurization, the proposed method yielded significant accuracy improvements, particularly for predicting the heat of formation in perovskites, reducing the error by up to 44.2% compared to default MODNet settings.

This approach was further generalized to more complex tasks such as convex hull distance and band gap prediction using halogen-containing materials from the OQMD dataset. SHAP plots and surrogate models were utilized to restore interpretability of the GNN features, extracting valuable chemical insights.

## Repository Contents
This repository includes:
- **Results**: The results obtained in the paper organized in the order they are presented.
  - `results/`
  
- **Datasets and models**: These are downloaded from [FigShare 
link](https://figshare.com/ndownloader/articles/27132093?private_link=ad92db8097ddc8d901f5), a python script is provided
to help with this.

## Getting Started
To replicate the results presented in the paper or to apply these methods to your own dataset, check the repository with the (modified MODNet version)[https://github.com/rogeriog/modnet_gnn_enhanced_v0.2.1] enhanced with the GNN featurizers, and check the scripts provided along with the results.

## Results
The main results of this work include:
- **Accuracy improvement**: The proposed method reduces the MAE for heat of formation predictions by 44.2% compared to the default MODNet featurizer.
- **Task generalization**: The method was successfully generalized to predict complex properties such as the convex hull distance and band gaps in halogen-containing materials from the OQMD dataset.
- **Interpretability**: Feature importance analysis with SHAP and surrogate models recovers chemical insights, enhancing the interpretability of GNN features.

## Contact and Contributions
For any clarifications, please contact the corresponding authors:  
- Gian-Marco Rignanese: [gian-marco.rignanese@uclouvain.be](mailto:gian-marco.rignanese@uclouvain.be)  
- Marcos José Leite Santos: [mjls@ufrgs.br](mailto:mjls@ufrgs.br)
