#!/bin/bash

#SBATCH --job-name=r400_2
#SBATCH --time=2:00:00
#SBATCH --mem=16000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#--gres=gpu:v100:1
#--gres=gpu:1

echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /data/p288722/python_venv/scd_images/bin/activate

nvidia-smi

# This flow is to generate results for prediction on varying number of patches
python /home/p288722/git_code/scd_images/run_flow_5.py -fold 2 -num_patches 400 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"

echo jobs completed
