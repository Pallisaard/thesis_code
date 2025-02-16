#!/bin/bash
#SBATCH --job-name=apply_mask_to_mris
#SBATCH --output=slurm_apply_mask_to_mris-%j.out # Name of output file
#SBATCH --error=slurm_apply_mask_to_mris-%j.err # Name of error file
#SBATCH --cpus-per-task=2  # Number of CPUs for each gpu
#SBATCH --mem=8G          # Memory request
# #SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=rpa@di.ku.dk # Email

cd ~/projects/thesis/thesis-code/

source .venv/bin/activate

python -m thesis_code.scripts.apply_mask_and_reorient --data-dir ../data/pre-training/not-brain-masked/ --output-dir ../data/pre-training/brain-masked-process-ready-resampled/ --fastsurfer-output-dir ../fastsurfer-output/pre-training/ --use-splits


# --percent-outliers 0.005