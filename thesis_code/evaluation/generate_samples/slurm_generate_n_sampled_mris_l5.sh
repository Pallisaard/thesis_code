#!/bin/bash
#SBATCH --job-name=generate_n_sampled_mris_l5
#SBATCH --output=slurm_generate_n_sampled_mris_l5-%j.out # Name of output file
#SBATCH --error=slurm_generate_n_sampled_mris_l5-%j.err # Name of error file
#SBATCH --gres=gpu:titanrtx:1       # Request 4 GPU per job
#SBATCH --time=01:15:00    # Time limit hrs:min:sec
#SBATCH --cpus-per-task=4  # Number of CPUs for each gpu
#SBATCH --mem=32G        # Memory request

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/generated-examples-lambda-5 \
                --n-samples 512 \
                --use-dp-safe \
                --checkpoint-path ../checkpoints/pretrain/all-data/HAGAN_l5_320k.ckpt \
                --lambdas 5 \
                --devices auto \
                --batch-size 2 \
                --vecotrizer-dim 512 \
