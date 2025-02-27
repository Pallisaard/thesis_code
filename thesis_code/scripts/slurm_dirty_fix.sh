#!/bin/bash
#SBATCH --job-name=dirty_fix
#SBATCH --output=dirty_fix-%j.out # Name of output file
#SBATCH --error=dirty_fix-%j.err # Name of error file
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16  # Number of CPUs for each gpu
#SBATCH --mem=8G          # Memory request
# #SBATCH --mail-type=END    # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=rpa@di.ku.dk # Email

cd ~/projects/thesis/thesis-code/

source .venv/bin/activate

python -m thesis_code.scripts.dirty_fix ../data/pre-training/brain-masked-process-ready-resampled/ --n-workers 14
