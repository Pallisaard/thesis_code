#!/bin/bash
#SBATCH --job-name=measure_quality_ssim_l5_512 # Name of the job
#SBATCH --output=measure_quality_ssim_l5_512-%j.out # Name of output file
#SBATCH --error=measure_quality_ssim_l5_512-%j.err # Name of error file
#SBATCH --time 02:00:00 # Runtime of the job
#SBATCH --gres=gpu:titanrtx:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=2  # Number of CPUs for each gpu
#SBATCH --mem=32G        # Memory request

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

python -m thesis_code.evaluation.quality_ssim.measure_quality_ssim --data-dir ../data/pre-training/brain-masked \
                --output-dir ../torch-output/pretrain-eval/quality-ssim \
                --checkpoint-path ../checkpoints/hagan_l5_320000/last.ckpt \
                --device 'cuda' \
                --n-samples 512 \
                --vector-dim 512 \
                --vectorizer-file ../torch-output/pretrain-eval/'true-examples-all'/mri_vectorizer_all_512_out.npy \
                --filename-file ../torch-output/pretrain-eval/'true-examples-all'/filenames.txt \
                --lambdas 1.0
