#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 11
#SBATCH --partition=normal
python train.py