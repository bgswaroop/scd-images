#!/bin/bash

#SBATCH --job-name=200
#SBATCH --time=48:00:00
#SBATCH --mem=4000
#SBATCH --cpus-per-task=1
# --partition=gpu
# --gres=gpu:v100:1
# --gres=gpu:k40:1
# --gres=gpu:1

echo starting_jobscript
module add CUDA/10.1.243-GCC-8.3.0

#tar -C /data/p288722/dresden/source_devices/tar_test_folder -cf /data/p288722/dresden/source_devices/tar_test_folder.tar .
tar -C /data/p288722/dresden/source_devices/nat_patches_128x128_200 -cf /data/p288722/dresden/source_devices/nat_patches_128x128_200.tar .

echo jobs completed

