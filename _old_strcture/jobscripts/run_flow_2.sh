#!/bin/bash

#SBATCH --job-name=th_pr
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --array=1-5


echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /data/p288722/python_venv/scd_images/bin/activate

nvidia-smi

# This flow is to generate results for prediction on varying number of patches
python /home/p288722/git_code/scd_images/run_flow_2.py -fold ${SLURM_ARRAY_TASK_ID} -num_patches 200 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "train_homo_test_random"


echo jobs completed
