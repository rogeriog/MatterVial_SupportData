#!/bin/bash
#
#
#SBATCH --job-name=featurize_mv           # Name your job
#SBATCH --output=%A_featurize_mv.out    # Output file (array job ID and task ID appended)
#SBATCH --partition=gpu                  
#SBATCH --nodes=1                                # Request one node
#SBATCH --ntasks-per-node=1                      # One task (serial job)
#SBATCH --mem=120G                               # Total memory on the node (adjust if needed)
#SBATCH --gpus=1  

## SBATCH --nodes=1                                
## SBATCH --ntasks-per-node=1
## SBATCH --cpus-per-task=32         
## SBATCH --mem=90G                               
#SBATCH --time=04:00:00                              
#SBATCH --account=htforft                        
#SBATCH --array=0          
echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"


# Activate your conda environment (adjust the path and environment name as needed)
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh
conda activate ML39

# Load the CUDA module (adjust version if needed)
module purge
module load EasyBuild/2022a CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python3 -u mattervial_featurizer_noorb.py



echo "Job finished on $(date)"