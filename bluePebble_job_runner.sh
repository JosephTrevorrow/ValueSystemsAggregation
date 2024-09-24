#!/bin/bash

#SBATCH --job-name=ia23938-iai-cdt-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:0:30
#SBATCH --mem=100M
#SBATCH --account=cosc023424

PYTHON_FILE="user/home/ia23938/ValueSystemsAggregation/experiment_runner.py"

python3 $PYTHON_FILE