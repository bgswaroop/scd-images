#!/bin/bash
#SBATCH --job-name=extract_patches
#SBATCH --time=6:00:00
#SBATCH --mem=16g
#SBATCH --ntasks=1
#--array=6-8,10-45,47-73
#SBATCH --array=18-22,28,29,38-41,66,67
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/scd-images/bin/activate

python /home/p288722/git_code/scd-images/project/data_modules/utils/extract_and_save_homo_patches.py \
  --patch_dims 128 --device_id "${SLURM_ARRAY_TASK_ID}" --num_patches 400


