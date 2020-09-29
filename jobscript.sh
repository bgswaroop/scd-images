#!/bin/bash

#SBATCH --job-name=dres_nat
#SBATCH --time=2:00:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
# --gres=gpu:k40:1
# --gres=gpu:1

echo starting_jobscript
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

echo activating environment
source /home/p288722/git_code/scd-autoencoders/venv/bin/activate

python run_flow.py
echo jobs completed
