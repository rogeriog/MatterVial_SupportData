#!/bin/bash
#SBATCH --job-name=adjacent_w_embed
#SBATCH --time=0-06:00:00
#SBATCH --output=MEGNetTest.txt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
module load CUDA cuDNN/8.0.4.30-CUDA-11.1.1 
# TensorFlow/2.5.0-fosscuda-2020b
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/ucl/modl/rgouvea/anaconda3/envs/env_tfmodnet/lib/"
CUDA_DIR=/home/ucl/modl/rgouvea/anaconda3/envs/env_tfmodnet/lib/python3.8/site-packages/nvidia/cuda_nvcc/


conda activate env_tfmodnet
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
# python3 -u generate_species.py COMBINE_UNRELAXED_DFS > log_unrelaxreduce.txt
python3 -u adjacent_training.py > log_adjacent_training.txt
echo "done"
date
