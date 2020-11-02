#!/bin/bash

#SBATCH --job-name=data_gen
#SBATCH --time=6:00:00
#SBATCH --mem=2000
#SBATCH --cpus-per-task=2
# --partition=gpu
# --gres=gpu:v100:1
# --gres=gpu:k40:1
# --gres=gpu:1

echo starting_jobscript
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

echo activating environment
source /home/p288722/git_code/scd-autoencoders/venv/bin/activate

python /home/p288722/git_code/scd-autoencoders/miscellaneous/convert_simlinks_to_txt.py
echo jobs completed
