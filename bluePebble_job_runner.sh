#!/bin/bash

#SBATCH --job-name=ia23938-iai-cdt-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:0:0
#SBATCH --mem=64G
#SBATCH --account=COSC023424

# Define the Python file you want to run
PYTHON_FILE="/home/ia23938/Documents/GitHub/ValueSystemsAggregation/experiment_runner.py"

python3 $PYTHON_FILE