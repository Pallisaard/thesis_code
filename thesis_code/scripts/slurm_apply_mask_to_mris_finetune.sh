#!/bin/bash
#SBATCH --job-name=apply_mask_to_mris_finetune
#SBATCH --output=slurm_apply_mask_to_mris_finetune-%j.out # Name of output file
#SBATCH --error=slurm_apply_mask_to_mris_finetune-%j.err # Name of error file
#SBATCH --cpus-per-task=2  # Number of CPUs for each gpu
#SBATCH --mem=8G          # Memory request
# #SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=rpa@di.ku.dk # Email

cd ~/projects/thesis/thesis-code/

source .venv/bin/activate

python -m thesis_code.data_collection.apply_mask_to_mris --data-dir ../data/finetune/not-brain-masked/ --fastsurfer-output-dir ../fastsurfer-output/finetune/

# --percent-outliers 0.005