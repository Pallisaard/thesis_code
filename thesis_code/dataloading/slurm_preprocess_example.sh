#!/bin/bash
#SBATCH --job-name=preprocess_example
#SBATCH --output=slurm_fastsurfer_%A_%a.out
#SBATCH --error=slurm_fastsurfer_%A_%a.err
#SBATCH --array=1-3%3   # Array job for 2740 MRI files, limit to 5 jobs running at once
#SBATCH --cpus-per-task=2  # Number of CPUs for each task
#SBATCH --mem=4G

cd ~/projects/thesis/thesis-code

source .venv/bin/activate

# If SLURM_ARRAY_TASK_ID is 1, set $dir to "train"
# else if SLURM_ARRAY_TASK_ID is 2, set $dir to "val"
# else if SLURM_ARRAY_TASK_ID is 3, set $dir to "test"
# else, echo "Invalid SLURM_ARRAY_TASK_ID" and exit
dir=""
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    dir="train"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    dir="val"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    dir="test"
else
    echo "Invalid SLURM_ARRAY_TASK_ID"
    exit
fi

echo "dir: "$dir

preprocess_dir="../data/pre-training/brain-masked/"$dir

echo "preprocess_dir: " $preprocess_dir

python -m thesis_code.dataloading.preprocess_example --nii-path $preprocess_dir --out-path $preprocess_dir --preprocess-folder --percent-outliers 0.999