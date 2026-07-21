#!/bin/bash
#SBATCH --job-name=ebola_90d_figures
#SBATCH --account=jandr
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --array=1-50
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=batch

echo "Starting SLURM task $SLURM_ARRAY_TASK_ID on $(hostname)..."
cd /home/groups/jandr/bundibugyo-ebov-vaccination/scripts || cd ~/bundibugyo-ebov-vaccination/scripts
python3 generate_fig3_spaghetti_90d.py --array-id $SLURM_ARRAY_TASK_ID
python3 generate_fig4_contours_90d.py --array-id $SLURM_ARRAY_TASK_ID
python3 generate_fig5_timing_dose_90d.py --array-id $SLURM_ARRAY_TASK_ID
