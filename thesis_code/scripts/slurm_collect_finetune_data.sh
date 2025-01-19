#!/bin/bash

#SBATCH --job-name=collect-finetune-data # Job name
#SBATCH --output=slurm_collect_finetune_data-%j.out # Name of output file
#SBATCH --error=slurm_collect_finetune_data-%j.err # Name of error file
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=2 # Number of CPU cores per task
#SBATCH --time=6:00:00 # Wall time
#SBATCH --mem-per-cpu=16000 # Memory per CPU core
#SBATCH --mail-user=rpa@di.ku.dk # Email
#SBATCH --mail-type=END # When to email

cd ~/home/projects/thesis/thesis-code
conda activate thesis
bash thesis_code/scripts/collect_finetune_data.sh ../data/fine-tuning
