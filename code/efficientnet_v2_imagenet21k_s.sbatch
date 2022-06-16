#!/bin/bash
#SBATCH --job-name keras-bert-normal
#SBATCH --output code/%x-%j.log
#SBATCH --error code/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv ~/SIF-images/tensorflow_latest-gpu-py3.sif python efficientnet_v2_imagenet21k_s.py
date
