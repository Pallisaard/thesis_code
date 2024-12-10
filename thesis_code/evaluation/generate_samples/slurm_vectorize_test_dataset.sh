#!/bin/bash
#SBATCH --job-name=vectorize_test_dataset
#SBATCH --output=slurm_vectorize_test_dataset-%j.out # Name of output file
#SBATCH --error=slurm_vectorize_test_dataset-%j.err # Name of error file
#SBATCH --gres=gpu:titanrtx:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=4  # Number of CPUs for each gpu
#SBATCH --mem=32G        # Memory request

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

python -m thesis_code.evaluation.generate_samples.vectorize_test_dataset --data-dir ../data/pre-training/brain-masked/train \
                --output-dir ../torch-output/pretrain-eval/'true-examples' \
                --device auto \
                --make-filename-file \
                --out-vectorizer-name mri_vectorizer_test_512_out.npy \
                --vectorizer-size 512 \

