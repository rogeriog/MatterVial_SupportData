#!/bin/bash
#SBATCH --job-name=perovsk5knoOFM_MODNet_BENCH
#SBATCH --time=100:00:00
#SBATCH --output=log.txt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40000
source ~/.bashrc

conda activate env_modnet
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
nproc=24 # $(nproc --all)
python3 run_benchmark.py --task matbench_perovskites --n_jobs $nproc >> log.txt
echo "done"
date
