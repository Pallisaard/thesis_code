#!/bin/bash

#SBATCH --job-name=collect-pretrain-data # Job name
#SBATCH --output=slurm_collect_pretrain_data-%j.out # Name of output file
#SBATCH --error=slurm_collect_pretrain_data-%j.err # Name of error file
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=12 # Number of CPU cores per task
#SBATCH --time=1-00:00:00 # Wall time
#SBATCH --mem-per-cpu=8000 # Memory per CPU core
#SBATCH --mail-user=rpa@di.ku.dk # Email
#SBATCH --mail-type=END # When to email

module load miniconda/24.5.0

cd ~/projects/thesis/thesis-code || exit 1

conda activate thesis || exit 1
# source .venv/bin/activate || exit 1

bash thesis_code/scripts/collect_data.sh ../data/pre-training/collections/
