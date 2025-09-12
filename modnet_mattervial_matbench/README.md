# Benchmarking MODNet@MatterVial on Matbench v0.1

This folder contains benchmark data for the [MODNet package](https://github.com/ppdebreuck/modnet) run on [Matbench v0.1]([200~https://hackingmaterials.lbl.gov/automatminer/datasets.html) datasets with [MatterVial featurizer](https://github.com/rogeriog/MatterVial). 

Full details can be found in the following paper:
> *Combining feature-based approaches with graph neural networks and symbolic regression for synergistic performance and interpretability* 
> Rogério Almeida Gouvêa, Pierre-Paul De Breuck, Tatiane Pretto, Gian-Marco Rignanese, Marcos José Leite Santos  
> (https://arxiv.org/abs/2509.03547).

Pre-computed or cached data will be used where possible, pending full upload of models and featurized dataframes to figshare. This repository currently contains *some* precomputed data (hence the ~400 MB size) but this will be moved to Figshare (and removed from the git history) in the future.

The reported benchmark results can be found in the `results` subfolder for each task as pickled Python dictionary with associated plots in the `plots` subfolders.

Results table:

<p align='center'>
<img src="../.github/Table1MatterVial.png" alt="benchmark results">
</p>


Results table MatBench v0.1 Perovskites:

<p align='center'>
<img src="../.github/Table2MatterVial.png" alt="benchmark results">
</p>
