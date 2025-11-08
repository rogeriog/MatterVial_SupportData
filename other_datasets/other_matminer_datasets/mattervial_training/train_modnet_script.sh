#!/bin/bash
#
#
#SBATCH --job-name=modnet_mv_training           # Name your job
#SBATCH --output=%A_%a_modnet_mv_training.out    # Output file (array job ID and task ID appended)
#SBATCH --partition=shared                   
#SBATCH --nodes=1                                
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64         
#SBATCH --mem=100G                               
#SBATCH --time=12:00:00                              
#SBATCH --account=htforft 
#SBATCH --array=0-5                       # Array job with 6 tasks (0 to 5)
echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"


# Activate your conda environment (adjust the path and environment name as needed)
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh
conda activate modnet2020

# Load the CUDA module (adjust version if needed)
module purge
module load EasyBuild/2022a CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0

# Define feature set combinations (one entry per array task)
declare -a feature_combinations=(
   "roost_all megnet_all mvl_all orb_v3"
   "roost_all megnet_all mvl_all orb_v3"
   "roost_all megnet_all mvl_all orb_v3"
   "roost_all megnet_all mvl_all orb_v3"
   "roost_all megnet_all mvl_all orb_v3"
   "roost_all megnet_all mvl_all orb_v3"
#    "matminer mvl32"
#    "matminer mvl_all"
#    ...
)

# Define corresponding data paths and suffixes (same index order as feature_combinations)
declare -a data_paths=(
  "boltztrap_mp_8924_mattervial_features.csv"
  "double_perovskites_gap_1306_mattervial_features.csv"
  "double_perovskites_gap_1306_optimized_mattervial_features.csv"
  "m2ax_223_mattervial_features.csv"
  "tholander_nitrides_12815_mattervial_features.csv"
  "wolverton_oxides_4914_mattervial_features.csv"
)

declare -a suffixes=(
   "boltztrap_mp_8924"
   "double_perovskites_gap_1306"
  "double_perovskites_gap_1306_optimized"
  "m2ax_223"
  "tholander_nitrides_12815"
  "wolverton_oxides_4914"

)

# Default to 0 when SLURM_ARRAY_TASK_ID is not set (for testing)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Basic validation
num_features=${#feature_combinations[@]}
num_data=${#data_paths[@]}
num_suffix=${#suffixes[@]}

if (( TASK_ID < 0 )) || (( TASK_ID >= num_features )); then
  echo "Error: SLURM_ARRAY_TASK_ID ($TASK_ID) out of range for feature_combinations (0..$((num_features-1)))"
  exit 1
fi
if (( num_data != num_features )) || (( num_suffix != num_features )); then
  echo "Error: Arrays feature_combinations, data_paths and suffixes must have the same length."
  echo "Lengths: feature_combinations=$num_features, data_paths=$num_data, suffixes=$num_suffix"
  exit 1
fi

# Select values for this array task
selected_features="${feature_combinations[$TASK_ID]}"
data_path="${data_paths[$TASK_ID]}"
suffix="${suffixes[$TASK_ID]}"

echo "Task ID: $TASK_ID"
echo "Selected features: ${selected_features}"
echo "Data path: ${data_path}"
echo "Suffix: ${suffix}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1


python3 -u masterscript_modnet.py MODNET_TRAIN \
    --data_path ${data_path} \
    --job_prefix "${suffix}_${selected_features// /_}" \
    --feature_sets ${selected_features} \
    --matbench_set ${suffix} \
    --n_jobs 64 \
    --hp_strategy ga 

echo "Job finished on $(date)"