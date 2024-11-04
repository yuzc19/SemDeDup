#!/bin/bash
#SBATCH --job-name=pipeline_job          # Job name
#SBATCH --output=log/pipeline_job.out        # Standard output and error log
#SBATCH --error=log/pipeline_job.err         # Error log
#SBATCH --partition=general       # Partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --gres=gpu:0                     # Number of GPUs per node
#SBATCH --mem=500G                        # Memory per node
#SBATCH --time=6:00:00                  # Time limit hrs:min:sec
#SBATCH --mail-type=BEGIN,END,FAIL       # Send email on job start, end, and failure
#SBATCH --mail-user=emilyx@andrew.cmu.edu  # Email address to send notifications


# Run the Python script
python pipeline.py

## USAGE
# conda activate myenv
# cd SemDeDup
# sbatch pipeline.sh


# Job ID: 95466
# Cluster: babel
# User/Group: emilyx/emilyx
# State: COMPLETED (exit code 0)
# Nodes: 1
# Cores per node: 16
# CPU Utilized: 01:28:23
# CPU Efficiency: 4.98% of 1-05:33:36 core-walltime
# Job Wall-clock time: 01:50:51
# Memory Utilized: 28.30 GB