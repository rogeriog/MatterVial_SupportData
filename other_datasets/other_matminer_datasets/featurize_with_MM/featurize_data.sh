#!/bin/bash
#
# Job-array submission script for running multiple independent featurization jobs
#
#SBATCH --job-name=multi_featurize              # A more general job name
#SBATCH --output=%A_%a_featurize.out            # Output file: %A is job ID, %a is array task ID
#SBATCH --partition=debug                      # Target partition (adjust as needed)
#SBATCH --nodes=1                               # Request one node
#SBATCH --ntasks-per-node=1                     # Serial job (one task per node)
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --mem=40G                               # Memory requested
#SBATCH --time=2:00:00                          # Maximum walltime
#SBATCH --account=htforft                       # Replace with your actual project/account name
#SBATCH --array=2                             # Array with two tasks (0 and 1) for your two commands

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Executing Task ID: $SLURM_ARRAY_TASK_ID"

# --- Environment Setup ---
module purge
module load EasyBuild/2022a CUDA/11.7.0

# Activate your conda environment
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh
conda activate modnet2020

# Move to the project directory

# --- Task-Specific Logic ---
# Use a 'case' statement to determine which command to run based on the task ID.

case $SLURM_ARRAY_TASK_ID in
  0)
    # Task 0: Featurize the m2ax dataset
    echo "Running command for m2ax_223.csv"
    DATA_FILE="./m2ax_223.csv"
    ;;
  1)
    # Task 1: Featurize the double perovskites dataset
    echo "Running command for double_perovskites_gap_1306.csv"
    DATA_FILE="./double_perovskites_gap_1306.csv"
    ;;
  2)
    # Task 2: Featurize the double perovskites dataset
    echo "Running command for double_perovskites_gap_1306_optimized.csv"
    DATA_FILE="./double_perovskites_gap_1306_optimized.csv"
    ;;
  *)
    echo "Error: Invalid Task ID ($SLURM_ARRAY_TASK_ID). Exiting."
    exit 1
    ;;
esac

# --- Execution ---
# Run the Python script with the parameters defined above.
# srun ensures proper resource allocation by Slurm.
# srun python3 -u masterscript_modnet.py MODNET_FEATURIZE \
#     --data_path "${DATA_FILE}" \
#     --featurizer_type "structure" \
#     --split_dataset 20
srun python3 -u masterscript_modnet.py MODNET_FEATURIZE \
    --data_path "${DATA_FILE}" \
    --featurizer_type "structure" \
    --split_dataset 20 \
    --start_chunk_index 19 \
    --end_chunk_index 20 

echo "Job finished on $(date)"