#!/bin/bash
NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN ./train.py hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled 
