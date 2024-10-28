#!/bin/bash
#SBATCH --job-name=slurm_pretrain_vae_64
#SBATCH --gres=gpu:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=2  # Number of CPUs for each gpu
#SBATCH --mem=16G          # Memory request
#SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email

module load cuda/11.8
module load cudnn/8.6.0

source ~/venv/bin/activate

cd ~/thesis_code

python -m thesis_code.training.pre_training.pretrain --model-name "cicek_3d_vae_64" \
                --latent-dim 1024 \
                --data-dir "~/final_dataset/fs_scans" \
                --batch-size 8 \
                --num-workers 0 \
                --transforms "resize" "range-normalize" \
                --resize-size 64 \
                --normalize-min 0 \
                --normalize-max 1 \
                --accelerator 'gpu' \
                --strategy 'ddp' \
                --devices 'auto' \
                --max-epochs 100 \
                --callbacks "checkpoint" "summary" "progress" \
                --checkpoint-dir "~/lightning/checkpoints" \
                --save-top-k 3 \
                --save-last \
                --fast-dev-run \