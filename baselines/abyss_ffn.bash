#!/bin/bash
#
# abyss_ffn.bash
#
#SBATCH --job-name=campfire_baseline_ffn      # Job name
#SBATCH --partition=htc                # Partition name
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --mem=1gb                      # Job Memory
#SBATCH --time=00:05:00                # Time limit hrs:min:sec
#SBATCH --output=./slurm_logs/array_%A-%a.log       # Standard output and error log
#SBATCH --array=0-232                # Array range 
 
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
 
echo "This is Task $SLURM_ARRAY_TASK_ID"
conda activate campfire
cd ~/campfire/baselines
python run_ffn.py expanded_gt.pkl --em-dir=./em --output-dir=./seg_out --row=$SLURM_ARRAY_TASK_ID
