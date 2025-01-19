#!/bin/bash
#SBATCH --job-name=preprocess_example
#SBATCH --output=slurm_preprocess_example_%j.out
#SBATCH --error=slurm_preprocess_example_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=2  # Number of CPUs for each task
#SBATCH --mem=4G

cd ~/projects/thesis/thesis-code

source .venv/bin/activate

dir="all"

echo "dir: "$dir

preprocess_dir="../data/fine-tuning/brain-masked-zerosliced-64/"$dir

echo "preprocess_dir: " $preprocess_dir

python -m thesis_code.scripts.preprocess_example --nii-path $preprocess_dir \
    --out-path $preprocess_dir \
    --preprocess-folder \
    --percent-outliers 0.999 \
    --size 64