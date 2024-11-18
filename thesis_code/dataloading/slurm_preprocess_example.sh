#!/bin/bash
#SBATCH --job-name=preprocess_example
#SBATCH --output=fastsurfer_%A_%a.out
#SBATCH --error=fastsurfer_%A_%a.err
#SBATCH --array=1-3%3   # Array job for 2740 MRI files, limit to 5 jobs running at once
# #SBATCH --gres=gpu:1       # Request 1 GPU per job
#SBATCH --cpus-per-task=2  # Number of CPUs for each task
# #SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=rpa@di.ku.dk # Email

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

preprocess_dir = ../data/pre-training/brain-masked/$dir

python -m thesis_code.dataloading.preprocess --input_dir $preprocess_dir --output_dir $preprocess_dir --process-folder --test