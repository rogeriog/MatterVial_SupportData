#!/bin/bash
#SBATCH --job-name=MODNet_perovskites_MNetMM60p_MNetOFM20p_MEGNet32_Adj
#SBATCH --time=1-00:00:00
#SBATCH --output=log.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=keira
#SBATCH --mem-per-cpu=8000
source ~/.bashrc

conda activate env_tfmodnet
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
nproc=24 # $(nproc --all)
python3 run_benchmark.py --task matbench_perovskites --n_jobs $nproc >> log.txt
echo "done"
date
