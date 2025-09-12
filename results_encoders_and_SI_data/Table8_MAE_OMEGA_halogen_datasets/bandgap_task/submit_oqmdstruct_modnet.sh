#!/bin/bash
#SBATCH --job-name=train_oqmdstruct_gap
#SBATCH --time=1-10:00:00
#SBATCH --output=oqmdstruct_gap.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=keira
#SBATCH --mem-per-cpu=8000

source ~/.bashrc

conda activate env_tfmodnet
#export PYTHONUSERBASE=intentionally-disabled  ##it was loading local modnet...
echo "start"
date
python3 -u oqmd_gap_model.py > oqmdstruct_gap1.txt  

echo "done"
date
