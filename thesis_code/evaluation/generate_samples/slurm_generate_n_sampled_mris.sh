#!/bin/bash
#SBATCH --job-name=generate_n_sampled_mris
#SBATCH --output=slurm_generate_n_sampled_mris-%j-%a.out # Name of output file
#SBATCH --error=slurm_generate_n_sampled_mris-%j-%a.err # Name of error file
#SBATCH --array=1-11%4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

echo "task id: $SLURM_ARRAY_TASK_ID"
echo 

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "HAGAN from authors"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/generated-examples-hagan-l5-orig \
                    --n-samples 1000 \
                    --checkpoint-path ../checkpoints/pretrained/hagan-l5-orig.ckpt \
                    --lambdas 5 \
                    --device auto \
                    --batch-size 2 \
                    --from-authors \
                    --vectorizer-dim 2048 \
                    --model-name hagan || { echo "Task 1 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo "HAGAN lambda 5-1"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/generated-examples-hagan-l5-1 \
                    --n-samples 1000 \
                    --use-dp-safe \
                    --checkpoint-path ../checkpoints/pretrained/hagan-l5-1.ckpt \
                    --lambdas 5 \
                    --device auto \
                    --batch-size 2 \
                    --vectorizer-dim 2048 \
                    --model-name hagan || { echo "Task 2 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    echo "HAGAN lambda 5-2"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/generated-examples-hagan-l5-2 \
                    --n-samples 1000 \
                    --use-dp-safe \
                    --checkpoint-path ../checkpoints/pretrained/hagan-l5-2.ckpt \
                    --lambdas 5 \
                    --device auto \
                    --batch-size 2 \
                    --skip-mri-save \
                    --vectorizer-dim 2048 \
                    --model-name hagan || { echo "Task 3 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    echo "HAGAN lambda 5-3"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/generated-examples-hagan-l5-3 \
                    --n-samples 1000 \
                    --use-dp-safe \
                    --checkpoint-path ../checkpoints/pretrained/hagan-l5-3.ckpt \
                    --lambdas 5 \
                    --device auto \
                    --batch-size 2 \
                    --skip-mri-save \
                    --vectorizer-dim 2048 \
                    --model-name hagan || { echo "Task 3 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    echo "HAGAN lambda 5-4"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/generated-examples-hagan-l5-4 \
                    --n-samples 1000 \
                    --use-dp-safe \
                    --checkpoint-path ../checkpoints/pretrained/hagan-l5-4.ckpt \
                    --lambdas 5 \
                    --device auto \
                    --batch-size 2 \
                    --skip-mri-save \
                    --vectorizer-dim 2048 \
                    --model-name hagan || { echo "Task 3 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    echo "HAGAN lambda 5-5"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/generated-examples-hagan-l5-5 \
                    --n-samples 1000 \
                    --use-dp-safe \
                    --checkpoint-path ../checkpoints/pretrained/hagan-l5-5.ckpt \
                    --lambdas 5 \
                    --device auto \
                    --batch-size 2 \
                    --skip-mri-save \
                    --vectorizer-dim 2048 \
                    --model-name hagan || { echo "Task 3 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    echo "WGAN-GP"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris \
                    --output-dir ../torch-output/pretrain-eval/generated-examples-wgan-gp \
                    --n-samples 1000 \
                    --checkpoint-path ../checkpoints/pretrained/wgan-gp.ckpt \
                    --device auto \
                    --batch-size 2 \
                    --vectorizer-dim 2048 \
                    --model-name wgan_gp || { echo "Task 4 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then
    echo "Alpha-GAN"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris \
                    --output-dir ../torch-output/pretrain-eval/generated-examples-alpha-gan \
                    --n-samples 1000 \
                    --checkpoint-path ../checkpoints/pretrained/alpha-gan.ckpt \
                    --device auto \
                    --batch-size 2 \
                    --vectorizer-dim 2048 \
                    --model-name alpha_gan || { echo "Task 5 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    echo "Kwon-GAN"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris \
                    --output-dir ../torch-output/pretrain-eval/generated-examples-kwon-gan \
                    --n-samples 1000 \
                    --checkpoint-path ../checkpoints/pretrained/kwon-gan.ckpt \
                    --device auto \
                    --batch-size 2 \
                    --vectorizer-dim 2048 \
                    --model-name kwon_gan || { echo "Task 6 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    echo "VAE-64"
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris \
                    --output-dir ../torch-output/pretrain-eval/generated-examples-vae-64 \
                    --n-samples 1000 \
                    --checkpoint-path ../checkpoints/pretrained/vae-64.ckpt \
                    --device auto \
                    --batch-size 2 \
                    --vectorizer-dim 2048 \
                    --model-name cicek_3d_vae_64 || { echo "Task 7 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 11 ]; then
    echo "Vectorizing test dataset"
    python -m thesis_code.evaluation.generate_samples.vectorize_test_dataset --data-dir ../data/pre-training/brain-masked-zerosliced \
                    --output-dir ../torch-output/pretrain-eval/'true-examples-all' \
                    --device 'cuda' \
                    --test-size 1000 \
                    --make-filename-file \
                    --vectorizer-dim 2048 || { echo "Task 8 failed"; exit 1; }

else
    echo "Invalid SLURM_ARRAY_TASK_ID"
    exit
fi
