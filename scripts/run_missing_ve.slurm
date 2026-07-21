#!/bin/bash
#SBATCH --job-name=ebola_missing
#SBATCH --account=jandr
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

module load python gcc

cd $SLURM_SUBMIT_DIR

source env/bin/activate
export PYTHONPATH=scripts

python3 run_missing_ve.py
