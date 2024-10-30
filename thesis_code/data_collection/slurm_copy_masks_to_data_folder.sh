#!/bin/bash
#SBATCH --job-name=copy-files
#SBATCH --output=copy-files-%j.out # Name of output file
#SBATCH --error=copy-files-%j.err # Name of error file
#SBATCH --cpus-per-task=2  # Number of CPUs for each gpu
#SBATCH --mem=8G          # Memory request
#SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email

source ~/venv/bin/activate

cd ~/thesis_code/

python -m thesis_code.data_collection.copy_masks_to_data_folder --data-dir ~/final_dataset --fastsurfer-output-dir ~/fastsurfer-output