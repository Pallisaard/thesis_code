#!/bin/bash
#SBATCH --job-name=copy-files
#SBATCH --output=copy-files-%j.out # Name of output file
#SBATCH --error=copy-files-%j.err # Name of error file
#SBATCH --cpus-per-task=2  # Number of CPUs for each gpu
#SBATCH --mem=8G          # Memory request
#SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email

cd ~/projects/thesis/thesis_code/

source .venv/bin/activate

python -m thesis_code.data_collection.apply_mask_to_mris --data-dir ../data/pre-training/ --fastsurfer-output-dir ~/fastsurfer-output