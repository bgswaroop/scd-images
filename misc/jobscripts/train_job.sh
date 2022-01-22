#!/bin/bash
#SBATCH --job-name=18m_flat_ps128
#SBATCH --time=2:00:00
#SBATCH --mem=64g
# --gres=gpu:v100:1
#SBATCH --partition=lab
#SBATCH --cpus-per-task=32
#SBATCH --array=0
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

# create the following directory manually
#SBATCH --chdir=/scratch/p288722/runtime_data/scd_images/scene_ind_test_set
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.out

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/scd-images/bin/activate

python /home/p288722/git_code/scd-images-lightning/project/main.py --fold "${SLURM_ARRAY_TASK_ID}" --dataset_dir="/scratch/p288722/datasets/dresden_new/nat_homo/patches_(128, 128)_400"
