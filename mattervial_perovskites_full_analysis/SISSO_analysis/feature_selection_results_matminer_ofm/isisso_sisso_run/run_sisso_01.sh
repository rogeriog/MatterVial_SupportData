#!/bin/bash
#
#SBATCH --job-name=sisso_run1             # Job name
#SBATCH --output=%j_run_sisso1.out                # Output file (%j appends the job ID)
#SBATCH --partition=shared                       
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=24              
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G                        
#SBATCH --time=18:00:00                   
#SBATCH --account=htforft                

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"

# Load necessary modules (adjust versions as needed)
module purge
module load EasyBuild/2024a OpenMPI/5.0.3-GCC-13.3.0 FlexiBLAS/3.4.4-GCC-13.3.0

# Activate your conda environment (change the path if necessary)
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh

# Change to the working directory where your scripts/data reside
###########################################################################
echo "Starting batch execution of sisso++"
# Activate conda environment for sisso run
conda activate sissopp_env


echo 'Entering folder: sisso_calc_target_isisso'
cd sisso_calc_target_isisso
LD_PRELOAD=/gpfs/softs/easybuild/2024a/software/GCCcore/13.3.0/lib64/libstdc++.so.6 mpiexec -np 24 sisso++ sisso.json
cd - > /dev/null

echo 'All tasks completed.'
