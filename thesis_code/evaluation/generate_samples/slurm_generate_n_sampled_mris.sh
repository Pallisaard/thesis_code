#!/bin/bash
#SBATCH --job-name=generate_n_sampled_mris_authors
#SBATCH --output=slurm_generate_n_sampled_mris_authors-%j.out # Name of output file
#SBATCH --error=slurm_generate_n_sampled_mris_authors-%j.err # Name of error file
#SBATCH --array=1-4%4   # Array job for 2740 MRI files, limit to 5 jobs running at once
#SBATCH --gres=gpu:a100:1       # Request 4 GPU per job
#SBATCH --time=01:15:00    # Time limit hrs:min:sec
#SBATCH --cpus-per-task=4  # Number of CPUs for each gpu
#SBATCH --mem=32G        # Memory request

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

echo "task id: $SLURM_ARRAY_TASK_ID"

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "This is the first task"
    sleep 3
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/small-generated-examples-authors \
                    --n-samples 512 \
                    --checkpoint-path ../checkpoints/pretrained/HAGAN_from_authors.ckpt \
                    --lambdas 5 \
                    --use-small-model \
                    --device auto \
                    --batch-size 2 \
                    --from-authors \
                    --vectorizer-dim 512 || { echo "Task 1 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo "This is the second task"
    sleep 3
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/small-generated-examples-lambda-1 \
                    --n-samples 512 \
                    --use-dp-safe \
                    --checkpoint-path ../checkpoints/pretrained/all-data/HAGAN_l1_320k.ckpt \
                    --lambdas 1 \
                    --use-small-model \
                    --device auto \
                    --batch-size 2 \
                    --vectorizer-dim 512  || { echo "Task 2 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    echo "This is the third task"
    sleep 3
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris --output-dir ../torch-output/pretrain-eval/small-generated-examples-lambda-5 \
                    --n-samples 512 \
                    --use-dp-safe \
                    --checkpoint-path ../checkpoints/pretrained/all-data/HAGAN_l5_320k.ckpt \
                    --lambdas 5 \
                    --use-small-model \
                    --device auto \
                    --batch-size 2 \
                    --vectorizer-dim 512 || { echo "Task 3 failed"; exit 1; }

elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    echo "This is the fourth task"
    sleep 3
    python -m thesis_code.evaluation.generate_samples.vectorize_test_dataset --data-dir ../data/pre-training/brain-masked \
                    --output-dir ../torch-output/pretrain-eval/'true-examples-all' \
                    --device 'cuda' \
                    --use-small-model \
                    --test-size 512 \
                    --make-filename-file \
                    --out-vectorizer-name mri_vectorizer_all_512_out.npy \
                    --vectorizer-dim 512 || { echo "Task 4 failed"; exit 1; }

else
    echo "Invalid SLURM_ARRAY_TASK_ID"
    exit
fi
