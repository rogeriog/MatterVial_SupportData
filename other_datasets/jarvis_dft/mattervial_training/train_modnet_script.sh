#!/bin/bash
#
#
#SBATCH --job-name=modnet_jarvis_training    # Name your job
#SBATCH --output=%A_%a_modnet_jarvis.out     # Output file (array job ID and task ID appended)
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --account=htforft
#
# IMPORTANT: The array range is set for the 29 target properties.
#SBATCH --array=1-10 
# -28

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"


# Activate your conda environment
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh
conda activate modnet2020

# Load the CUDA module
module purge
module load EasyBuild/2022a CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0

# --- CONFIGURATION ---

# Define target properties to iterate over (29 total)
declare -a target_names=(
    'p-powerfact' 'formation_energy_peratom' 'optb88vdw_bandgap' 'optb88vdw_total_energy'
    'ehull' 'mbj_bandgap' 'bulk_modulus_kv' 'shear_modulus_gv' 'magmom_oszicar'
    'slme' 'spillage' 'kpoint_length_unit' 'encut' 'epsx' 'epsy' 'epsz'
    'mepsx' 'mepsy' 'mepsz' 'dfpt_piezo_max_dielectric' 'dfpt_piezo_max_dij'
    'dfpt_piezo_max_eij' 'exfoliation_energy' 'max_efg' 'avg_elec_mass'
    'avg_hole_mass' 'n-Seebeck' 'n-powerfact' 'p-Seebeck'
)

# Set the static dataset and feature information
data_path="jarvis_dft_3d_mattervial_features.csv"
suffix="jarvis_dft_3d"
selected_features="roost_all megnet_all mvl_all orb_v3"


# --- JOB ARRAY LOGIC ---

# Default to 0 when SLURM_ARRAY_TASK_ID is not set (for local testing)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# The TASK_ID directly corresponds to the index in the target_names array
num_targets=${#target_names[@]}
if (( TASK_ID < 0 )) || (( TASK_ID >= num_targets )); then
    echo "Error: SLURM_ARRAY_TASK_ID ($TASK_ID) is out of range for target_names (0..$((num_targets-1)))"
    exit 1
fi

# Select the target property for this specific array task
target_name="${target_names[$TASK_ID]}"


# --- EXECUTION ---

echo "========================================================"
echo "SLURM Task ID: $TASK_ID"
echo "--------------------------------------------------------"
echo "  > Data path:       ${data_path}"
echo "  > Suffix:          ${suffix}"
echo "  > Selected features: ${selected_features}"
echo "  > Target Property: ${target_name}"
echo "========================================================"

# Set threading environment variables for performance
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

python3 -u masterscript_modnet.py MODNET_TRAIN \
    --data_path "${data_path}" \
    --job_prefix "${suffix}_${target_name}_${selected_features// /_}" \
    --feature_sets ${selected_features} \
    --matbench_set "${suffix}" \
    --target_name "${target_name}" \
    --n_jobs 64 \
    --hp_strategy ga

echo "Job finished on $(date)"