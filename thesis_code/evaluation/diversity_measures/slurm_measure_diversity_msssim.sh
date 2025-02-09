#!/bin/bash
#SBATCH --job-name=measure_msssim
#SBATCH --output=slurm_measure_msssim-%j-%a.out
#SBATCH --error=slurm_measure_msssim-%j-%a.err
#SBATCH --array=1-10%5
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=rpa@di.ku.dk

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

echo "task id: $SLURM_ARRAY_TASK_ID"
echo 

BASE_DIR="../torch-output/pretrain-eval"

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "HAGAN from authors"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-authors" \
        --device cuda \
        --resolution 256 \
        --output-file "$BASE_DIR/generated-examples-authors/msssim_scores.pt" || { echo "Task 1 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo "HAGAN lambda 5-1"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-hagan-l5-1" \
        --device cuda \
        --resolution 256 \
        --output-file "$BASE_DIR/generated-examples-hagan-l5-1/msssim_scores.pt" || { echo "Task 2 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    echo "HAGAN lambda 5-2"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-hagan-l5-2" \
        --device cuda \
        --resolution 256 \
        --output-file "$BASE_DIR/generated-examples-hagan-l5-2/msssim_scores.pt" || { echo "Task 3 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    echo "HAGAN lambda 5-3"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-hagan-l5-3" \
        --device cuda \
        --resolution 256 \
        --output-file "$BASE_DIR/generated-examples-hagan-l5-3/msssim_scores.pt" || { echo "Task 4 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    echo "HAGAN lambda 5-4"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-hagan-l5-4" \
        --device cuda \
        --resolution 256 \
        --output-file "$BASE_DIR/generated-examples-hagan-l5-4/msssim_scores.pt" || { echo "Task 5 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    echo "HAGAN lambda 5-5"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-hagan-l5-5" \
        --device cuda \
        --resolution 256 \
        --output-file "$BASE_DIR/generated-examples-hagan-l5-5/msssim_scores.pt" || { echo "Task 6 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    echo "WGAN-GP"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-wgan-gp" \
        --device cuda \
        --resolution 64 \
        --output-file "$BASE_DIR/generated-examples-wgan-gp/msssim_scores.pt" || { echo "Task 7 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 8 ]; then
    echo "Alpha-GAN"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-alpha-gan" \
        --device cuda \
        --resolution 64 \
        --output-file "$BASE_DIR/generated-examples-alpha-gan/msssim_scores.pt" || { echo "Task 8 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    echo "Kwon-GAN"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-kwon-gan" \
        --device cuda \
        --resolution 64 \
        --output-file "$BASE_DIR/generated-examples-kwon-gan/msssim_scores.pt" || { echo "Task 9 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    echo "VAE-64"
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "$BASE_DIR/generated-examples-vae-64" \
        --device cuda \
        --resolution 64 \
        --output-file "$BASE_DIR/generated-examples-vae-64/msssim_scores.pt" || { echo "Task 10 failed"; exit 1; }

else
    echo "Invalid SLURM_ARRAY_TASK_ID"
    exit 1
fi 