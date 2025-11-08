#!/bin/bash
#
#
#SBATCH --job-name=featurize_mv           # Name your job
#SBATCH --output=%A_%a_featurize_mv.out    # Output file (array job ID and task ID appended)
#SBATCH --partition=debug                  
#SBATCH --nodes=1                                
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32   # 4        
#SBATCH --mem=40G                               
#SBATCH --time=02:00:00                              
#SBATCH --account=htforft                        
#SBATCH --array=0          
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
python3 -u mattervial_featurizer_noorb.py --dataset_name alexandria5000test



echo "Job finished on $(date)"