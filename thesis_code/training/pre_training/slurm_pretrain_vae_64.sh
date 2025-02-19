#!/bin/bash
#SBATCH --job-name=pretrain_vae_64
#SBATCH --output=slurm_pretrain_vae_64-%j.out # Name of output file
#SBATCH --error=slurm_pretrain_vae_64-%j.err # Name of error file
#SBATCH --gres=gpu:l40s:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=8  # Number of CPUs for each gpu
#SBATCH --time=24:00:00    # Limit to 36 hours.
#SBATCH --mem=32G          # Memory request
#SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email
#SBATCH --dependency=afterany:5109_1

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

python -m thesis_code.training.pre_training.pretrain --model-name "cicek_3d_vae_64" \
                --latent-dim 1024 \
                --data-path ../data/pre-training/brain-masked-no-zerosliced-64 \
                --use-all-data-for-training \
                --batch-size 32 \
                --num-workers 6 \
                --resize-size 64 \
                --accelerator gpu \
                --devices auto \
                --max-steps 160000 \
                --callbacks checkpoint summary \
                --save-top-k 3 \
                --save-last \
                --log-every-n-steps 50 \