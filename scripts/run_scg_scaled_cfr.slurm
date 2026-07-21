#!/bin/bash
#SBATCH --job-name=ebola_scaled_cfr
#SBATCH --account=jandr
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-24

module load python gcc

cd $SLURM_SUBMIT_DIR

source ../env/bin/activate
export PYTHONPATH=../scripts

export SCG_SUBSET_NAME="scaled_cfr"
python3 run_scg_subset.py
