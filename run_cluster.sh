#!/bin/bash

srun -t 15000 --gpus=1 -p normal -c 11 train.sh TORCH_HOME=/home/martemev/vklyukin/weights/models
