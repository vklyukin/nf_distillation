#!/bin/bash

sbatch -t 15000 --gpus=1 -p normal -c 10 train.sh
