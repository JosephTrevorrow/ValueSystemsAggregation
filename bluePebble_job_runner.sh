#!/bin/bash

#SBATCH --job-name=ia23938-iai-cdt-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:0
#SBATCH --mem=1G
#SBATCH --account=cosc023424

PYTHON_FILE="experiment_runner.py"

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

module add gcc/12.3.0

source ~/ValueSystemsAggregation/initConda.sh

conda activate new_env

python -m pip install -r requirements.txt
python $PYTHON_FILE

