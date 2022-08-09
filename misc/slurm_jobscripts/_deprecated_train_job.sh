#!/bin/bash
#SBATCH --job-name=ps128_200_stddev
#SBATCH --time=15:40:00
#SBATCH --mem=64g
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --array=0-4
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

# create the following directory manually
#SBATCH --output=/scratch/p288722/runtime_data/scd_images/ps128_200_stddev/fold_%a/slurm-%A_%a.out
#SBATCH --error=/scratch/p288722/runtime_data/scd_images/ps128_200_stddev/fold_%a/slurm-%A_%a.out

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/scd-images/bin/activate

dataset_dir="/scratch/p288722/datasets/dresden_new/nat_homo_stddev/patches_(128, 128)_200"
default_root_dir="/scratch/p288722/runtime_data/scd_images/ps128_200_stddev/fold_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${default_root_dir}"
# shellcheck disable=SC2164
cd "${default_root_dir}"

classifier="all_brands"
python /home/p288722/git_code/scd-images/project/main.py --fold "${SLURM_ARRAY_TASK_ID}" --dataset_dir="${dataset_dir}" --default_root_dir="${default_root_dir}" --classifier=${classifier}
mv lightning_logs/version_"${SLURM_JOB_ID}" ./${classifier}

classifier="Nikon_models"
python /home/p288722/git_code/scd-images/project/main.py --fold "${SLURM_ARRAY_TASK_ID}" --dataset_dir="${dataset_dir}" --default_root_dir="${default_root_dir}" --classifier=${classifier}
mv lightning_logs/version_"${SLURM_JOB_ID}" ./${classifier}

classifier="Samsung_models"
python /home/p288722/git_code/scd-images/project/main.py --fold "${SLURM_ARRAY_TASK_ID}" --dataset_dir="${dataset_dir}" --default_root_dir="${default_root_dir}" --classifier=${classifier}
mv lightning_logs/version_"${SLURM_JOB_ID}" ./${classifier}

classifier="Sony_models"
python /home/p288722/git_code/scd-images/project/main.py --fold "${SLURM_ARRAY_TASK_ID}" --dataset_dir="${dataset_dir}" --default_root_dir="${default_root_dir}" --classifier=${classifier}
mv lightning_logs/version_"${SLURM_JOB_ID}" ./${classifier}

classifier="all_models"
python /home/p288722/git_code/scd-images/project/main.py --fold "${SLURM_ARRAY_TASK_ID}" --dataset_dir="${dataset_dir}" --default_root_dir="${default_root_dir}" --classifier=${classifier}
mv lightning_logs/version_"${SLURM_JOB_ID}" ./${classifier}
