#!/bin/bash
#SBATCH --job-name=pretrain_vae_64
#SBATCH --output=slurm_pretrain_vae_64-%j.out # Name of output file
#SBATCH --error=slurm_pretrain_vae_64-%j.err # Name of error file
#SBATCH --gres=gpu:a100:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=10  # Number of CPUs for each gpu
#SBATCH --mem=32G          # Memory request
#SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email

module load cuda/11.8
module load cudnn/8.6.0

source ~/venv/bin/activate

cd ~/thesis_code

python -m thesis_code.training.pre_training.pretrain --model-name "cicek_3d_vae_64" \
                --latent-dim 1024 \
                --data-path ../data/pre-training/brain-masked-zerosliced-64 \
                --use-all-data-for-training \
                --batch-size 32 \
                --num-workers 8 \
                --resize-size 64 \
                --accelerator gpu \
                --devices auto \
                --max-epochs 100 \
                --callbacks checkpoint summary \
                --save-top-k 3 \
                --save-last \
                --log-every-n-steps 50 \