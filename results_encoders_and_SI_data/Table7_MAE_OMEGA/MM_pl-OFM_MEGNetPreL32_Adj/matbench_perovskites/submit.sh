#!/bin/bash
#SBATCH --job-name=MODNet_perovskites_MNetencodOFM20p_MEGNet32
#SBATCH --time=1-00:00:00
#SBATCH --output=log.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=keira
#SBATCH --mem-per-cpu=4000
source ~/.bashrc

conda activate env_tfmodnet
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
nproc=32 # $(nproc --all)
python3 run_benchmark.py --task matbench_perovskites --n_jobs $nproc >> log.txt
echo "done"
date
