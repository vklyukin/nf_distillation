#!/bin/bash

#sbatch -t 07:00:00 --gpus=1 -p normal -c 4 train.sh

set -x
python train.py "$@"