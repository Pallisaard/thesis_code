#!/bin/bash
#SBATCH --job-name=finetune_vectorize_test_dataset
#SBATCH --output=finetune_vectorize_test_dataset-%j.out # Name of output file
#SBATCH --error=finetune_vectorize_test_dataset-%j.err # Name of error file
#SBATCH --gres=gpu:titanrtx:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=4  # Number of CPUs for each gpu
#SBATCH --mem=32G        # Memory request

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

python -m thesis_code.training.evaluation.vectorize_test_dataset --data-dir ../data/finetune/brain-masked/test \
                --output-dir ../torch-output/pretrain-eval/'ft-true-examples' \
                --device 'cuda' \
                --test-size 200 \
