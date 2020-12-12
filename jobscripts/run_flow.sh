#!/bin/bash

#SBATCH --job-name=5
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#--gres=gpu:1

echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /home/p288722/git_code/scd-autoencoders/venv/bin/activate

nvidia-smi

python /home/p288722/git_code/scd-autoencoders/run_flow.py -fold 5 -num_patches 100

echo jobs completed

