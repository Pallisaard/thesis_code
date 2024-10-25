#!/bin/bash
#SBATCH --job-name=slurm_pretrain_vae_64
#SBATCH --gres=gpu:4       # Request 4 GPU per job
#SBATCH --cpus_per_tasks=6 # Number of CPUs for each gpu
#SBATCH --mem=32G          # Memory request
#SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email

module load cuda/11.8
module load cudnn/8.6.0

source ~/venv/bin/activate

python pretrain --model-name "cicek-3d-vae" \
                --latent-dim 256 \
                --data-dir "~/final_dataset" \
                --batch-size 64 \
                --num-workers 4 \
                --transforms "resize" "range-normalize" \
                --resize-size 64 \