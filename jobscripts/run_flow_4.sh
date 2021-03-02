#!/bin/bash

#SBATCH --job-name=dev1
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#--gres=gpu:1

echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /data/p288722/python_venv/scd_images/bin/activate

nvidia-smi

# Patch Aggregation Choices: 'majority_vote', 'prediction_score_sum', 'log_scaled_prediction_score_sum', 'log_scaled_std_dev'

echo running job 1

# This flow runs the flat classifier (for devices)
python /home/p288722/git_code/scd_images/run_flow_4.py -fold 1 -num_patches 200 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "homo"

echo jobs completed
