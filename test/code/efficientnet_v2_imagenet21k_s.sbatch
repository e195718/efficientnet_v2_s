#!/bin/bash
#SBATCH --job-name keras-bert-normal
#SBATCH --output %x-%j.log
#SBATCH --error %x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv ~/efficientnetv2/efficientnet_v2_s/code/tensorflow_latest-gpu-py3.sif python efficientnet_v2_imagenet21k_s.py
date
