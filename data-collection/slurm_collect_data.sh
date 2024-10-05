#!/bin/bash

#SBATCH --job-name=gather-pretraining-data # Job name
#SBATCH --output=gather-pretraining-data-%j.out # Name of output file
#SBATCH --error=gather-pretraining-data-%j.err # Name of error file
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=4 # Number of CPU cores per task
#SBATCH --time=1-00:00:00 # Wall time
#SBATCH --mem-per-cpu=8000 # Memory per CPU core
#SBATCH --mail-user=rpa@di.ku.dk # Email
#SBATCH --mail-type=ALL # When to email

cd ~/home/projects/thesis/thesis-code
conda activate thesis
bash data-collection/collect_data.sh ../data
