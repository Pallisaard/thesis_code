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

python -m thesis_code.scripts.apply_mask_and_reorient --data-dir ../data/fine-tuning/fs_scans/ --output-dir ../data/fine-tuning/brain-masked --fastsurfer-output-dir ../fastsurfer-output/fine-tuning/
