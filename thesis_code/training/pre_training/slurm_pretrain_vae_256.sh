#!/bin/bash
#SBATCH --job-name=pretrain_vae_256
#SBATCH --output=slurm_pretrain_vae_256-%j.out # Name of output file
#SBATCH --error=slurm_pretrain_vae_256-%j.err # Name of error file
#SBATCH --gres=gpu:titanrtx:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=6  # Number of CPUs for each gpu
#SBATCH --mem=16G          # Memory request
# #SBATCH --mail-type=ALL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=rpa@di.ku.dk # Email

module load cuda/11.8
module load cudnn/8.6.0

source ~/venv/bin/activate

cd ~/thesis_code

python -m thesis_code.training.pre_training.pretrain --model-name "cicek_3d_vae_256" \
                --latent-dim 1024 \
                --data-path /home/gzj557/final_dataset \
                --batch-size 2 \
                --num-workers 5 \
                --transforms resize range-normalize \
                --resize-size 256 \
                --normalize-min 0 \
                --normalize-max 1 \
                --accelerator gpu \
                --strategy ddp \
                --devices auto \
                --max-epochs 100 \
                --callbacks checkpoint summary progress \
                --save-top-k 3 \
                --save-last \
                --fast-dev-run \