#!/bin/bash

#SBATCH --job-name=5
#SBATCH --time=4:00:00
#SBATCH --mem=8000
#SBATCH --cpus-per-task=12
#--partition=cpu
#--gres=gpu:v100:1
#--gres=gpu:1

echo starting_jobscript
module add OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
module add CUDA/10.1.243-GCC-8.3.0

echo activating environment
source /home/p288722/git_code/scd-autoencoders/venv/bin/activate

nvidia-smi

python run_flow_prnu.py -fold 5
echo jobs completed
