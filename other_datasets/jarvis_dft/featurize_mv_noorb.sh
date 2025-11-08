#!/bin/bash
#
#
#SBATCH --job-name=featurize_mv           # Name your job
#SBATCH --output=%A_%a_featurize_mv.out    # Output file (array job ID and task ID appended)
#SBATCH --partition=shared                  
##SBATCH --mem=120G                               # Total memory on the node (adjust if needed)
##SBATCH --gpus=1  
#SBATCH --nodes=1                                
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32         
#SBATCH --mem=90G                               
#SBATCH --time=05:00:00                              
#SBATCH --account=htforft                        
#SBATCH --array=1-5          
echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"


# Activate your conda environment (adjust the path and environment name as needed)
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh
conda activate ML39

# Load the CUDA module (adjust version if needed)
module purge
module load EasyBuild/2023a CUDA/12.2.0 cuDNN/8.9.2.26-CUDA-12.2.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python3 -u mattervial_featurizer_noorb.py --dataset_name jarvis_dft_3d --n_chunks 5 --chunk $SLURM_ARRAY_TASK_ID



echo "Job finished on $(date)"