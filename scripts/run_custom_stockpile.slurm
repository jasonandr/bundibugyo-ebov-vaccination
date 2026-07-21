#!/bin/bash
#SBATCH --job-name=ebola_scg_stockpile
#SBATCH --account=jandr
#SBATCH --partition=batch
#SBATCH --array=0-5
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

module load python gcc

cd $SLURM_SUBMIT_DIR

# We assume pybind11 and environment is already set up from earlier jobs
source env/bin/activate
export PYTHONPATH=scripts

python3 run_scg_custom_stockpile.py
