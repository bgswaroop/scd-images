#!/bin/bash

#SBATCH --job-name=data3
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --array=0-73

echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /data/p288722/python_venv/scd_images/bin/activate

which python

python /home/p288722/git_code/scd_images/miscellaneous/prepare_image_and_patch_data.py -task_num ${SLURM_ARRAY_TASK_ID}
#python /home/p288722/git_code/scd_images/miscellaneous/prepare_image_and_patch_data_2.py -task_num ${SLURM_ARRAY_TASK_ID}
#python /home/p288722/git_code/scd_images/miscellaneous/prepare_image_and_patch_data_3.py -task_num ${SLURM_ARRAY_TASK_ID}

echo jobs completed

