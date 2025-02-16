#!/bin/bash
#SBATCH --job-name=preprocess-example-64
#SBATCH --output=slurm_preprocess-example-64_%A_%a.out
#SBATCH --error=slurm_preprocess-example-64_%A_%a.err
#SBATCH --time=0:10:00
#SBATCH --array=1-3%3   # Array job for 2740 MRI files, limit to 5 jobs running at once
#SBATCH --cpus-per-task=10  # Number of CPUs for each task
#SBATCH --mem=12G

cd ~/projects/thesis/thesis-code

source .venv/bin/activate

# If SLURM_ARRAY_TASK_ID is 1, set $dir to "train"
# else if SLURM_ARRAY_TASK_ID is 2, set $dir to "val"
# else if SLURM_ARRAY_TASK_ID is 3, set $dir to "test"
# else, echo "Invalid SLURM_ARRAY_TASK_ID" and exit
dir=""
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    dir="train"
    n_workers=8
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    dir="val"
    n_workers=2
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    dir="test"
    n_workers=2
else
    echo "Invalid SLURM_ARRAY_TASK_ID"
    exit
fi

echo "dir: "$dir

preprocess_dir="../data/pre-training/brain-masked-zerosliced-64/"$dir

echo "preprocess_dir: " $preprocess_dir

python -m thesis_code.scripts.preprocess_example --nii-path $preprocess_dir \
    --out-path $preprocess_dir \
    --preprocess-folder \
    --percent-outliers 0.999 \
    --remove-zero-slices \
    --size 64 \
    --n-workers $n_workers