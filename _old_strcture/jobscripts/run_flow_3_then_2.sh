#!/bin/bash

#SBATCH --job-name=s4
#SBATCH --time=24:00:00
#SBATCH --mem=24000
#SBATCH --cpus-per-task=8
#SBATCH --partition=lab
#SBATCH --reservation=infsys
#--gres=gpu:v100:1
#SBATCH --gres=gpu:1

echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /data/p288722/python_venv/scd_images/bin/activate

nvidia-smi

# This flow can be used to check for convergence of all the models + brand classifier
python /home/p288722/git_code/scd_images/run_flow_3.py -fold 4 -num_patches 400 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_3.py -fold 5 -num_patches 400 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
															 
# This flow is used for testing with varying number of patches
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 1 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 5 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 10 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 20 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 40 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 100 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 200 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"
python /home/p288722/git_code/scd_images/run_flow_2.py -fold 4 -num_patches 400 -patch_aggregation "majority_vote" -use_contributing_patches 0 -patches_type "random"

echo jobs completed
