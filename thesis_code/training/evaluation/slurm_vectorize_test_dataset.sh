#!/bin/bash
#SBATCH --job-name=vectorize_test_dataset
#SBATCH --output=slurm_generate_n_sampled_mris-%j.out # Name of output file
#SBATCH --error=slurm_generate_n_sampled_mris-%j.err # Name of error file
#SBATCH --gres=gpu:titanrtx:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=4  # Number of CPUs for each gpu
#SBATCH --mem=32G        # Memory request

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

python -m thesis_code.training.evaluation.generate_n_sampled_mris --data-dir ../data/pre-training/brain-masked/test \
                --output-dir ../torch-output/'true-examples-pretrain' \
                --device 'cuda' \
