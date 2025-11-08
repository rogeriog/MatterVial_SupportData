#!/bin/bash
#
#
#SBATCH --job-name=cogn_train            # Name your job
#SBATCH --output=%j_congn_gpu.out          # Output file (job ID appended)
#SBATCH --partition=debug-gpu                     # Debug GPU partition for testing
#SBATCH --nodes=1                                # Request one node
#SBATCH --ntasks-per-node=1                      # One task (serial job)
#SBATCH --mem=100G                               # Total memory on the node (adjust if needed)
#SBATCH --gpus=1                                 # Request one GPU (change if necessary)
#SBATCH --time=2:00:00                           # Maximum walltime (2h for debug jobs)
#SBATCH --account=htforft                        # Replace with your actual project/account name

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"

# Load the CUDA module (adjust version if needed)
module purge
module load EasyBuild/2022a CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
# Activate your conda environment (adjust the path and environment name as needed)
# source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh
# conda activate kgcnn-env
source .venv/bin/activate
python3 -u process_coGN.py

echo "Job finished on $(date)"
