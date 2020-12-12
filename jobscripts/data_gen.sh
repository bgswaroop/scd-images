#!/bin/bash

#SBATCH --job-name=5
#SBATCH --time=03:00:00
#SBATCH --mem=1000
#SBATCH --cpus-per-task=1
# --partition=gpu
# --gres=gpu:v100:1
# --gres=gpu:k40:1
# --gres=gpu:1

echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /data/p288722/python_venv/scd_images/bin/activate

which python

python /home/p288722/git_code/scd_images/miscellaneous/prepare_image_and_patch_data.py
echo jobs completed

