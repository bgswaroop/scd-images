#!/bin/bash
#SBATCH --job-name=run_flow
#SBATCH --time=2:00:00
#SBATCH --mem=100g
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpushort
#SBATCH --cpus-per-task=12
#SBATCH --array=0 # change it to 0-4 to train on all 4 folds
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

# create the following directory manually
#SBATCH --output=/data/p288722/runtime_data/scd_images/reproduce_results/fold_%a/slurm-%A_%a.out
#SBATCH --error=/data/p288722/runtime_data/scd_images/reproduce_results/fold_%a/slurm-%A_%a.out

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/scd-images/bin/activate

patches_dataset_dir="/scratch/p288722/datasets/dresden/nat_homo/patches_(128, 128)_200"
full_image_dataset_dir="/data/p288722/datasets/dresden/source_devices/natural"
default_root_dir="/data/p288722/runtime_data/scd_images/reproduce_results/fold_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${default_root_dir}"
# shellcheck disable=SC2164
cd "${default_root_dir}"

classifier="all_brands"
python /home/p288722/git_code/scd-images/project/main.py --fold "${SLURM_ARRAY_TASK_ID}" --patches_dataset_dir="${patches_dataset_dir}" --full_image_dataset_dir="${full_image_dataset_dir}" --default_root_dir="${default_root_dir}" --classifier=${classifier}
mv lightning_logs/version_"${SLURM_JOB_ID}" ./${classifier}
