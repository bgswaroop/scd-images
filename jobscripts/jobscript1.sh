#!/bin/bash

#SBATCH --job-name=data_gen
#SBATCH --time=01:30:00
#SBATCH --mem=1000
#SBATCH --cpus-per-task=1
# --partition=gpu
# --gres=gpu:v100:1
# --gres=gpu:k40:1
# --gres=gpu:1

echo starting_jobscript
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

echo activating environment
source /home/p288722/git_code/scd-autoencoders/venv/bin/activate

python /home/p288722/git_code/scd-autoencoders/miscellaneous/prepare_image_and_patch_data.py
echo jobs completed
