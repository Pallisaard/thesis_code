#!/bin/bash

#SBATCH --job-name=gather-pretraining-data # Job name
#SBATCH --output=gather-pretraining-data-%j.out # Name of output file
#SBATCH --error=gather-pretraining-data-%j.err # Name of error file
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=6 # Number of CPU cores per task
#SBATCH --time=3-00:00:00 # Wall time
#SBATCH --mem-per-cpu=12000 # Memory per CPU core
#SBATCH -p gpu --gres=gpu:titanx:4
#SBATCH --mail-user=rpa@di.ku.dk # Email
#SBATCH --mail-type=ALL # When to email

cd ~/home/projects/thesis/thesis-code
conda activate thesis
bash training/slurm_pretrain_vae.sh \
  --model-name "cicek_3d_vae" \
  --latent-dim 256 \
  --data-dir "../data/final_dataset" \
  --batch-size 32 \
  --n-workers 4 \
  --normalize-dir ../data/z_score_params.txt \
  --accelerator "auto" \
  --strategy "ddp" \
  --devices "auto" \
  --fast-dev-run True \
  --max-epochs 300
