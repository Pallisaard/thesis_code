#!/bin/bash
#SBATCH --job-name=pretrain_vae_256
#SBATCH --output=slurm_pretrain_vae_256-%j.out # Name of output file
#SBATCH --error=slurm_pretrain_vae_256-%j.err # Name of error file
#SBATCH --gres=gpu:a100:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=10  # Number of CPUs for each gpu
#SBATCH --time=1-12:00:00    # Limit to 36 hours.
#SBATCH --mem=16G        # Memory request
#SBATCH --mail-type=END    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

# echo devices and nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# echo time at start
echo "start time: $(date)"

python -m thesis_code.training.pre_training.pretrain --model-name "cicek_3d_vae_256" \
                --latent-dim 1024 \
                --data-path ../data/pre-training/brain-masked  \
                --use-all-data-for-training \
                --batch-size 4 \
                --num-workers 8 \
                --accelerator gpu \
                --devices auto \
                --callbacks 'checkpoint' \
                --save-top-k 3 \
                --save-last \
                --log-every-n-steps 50 \
                --max-steps 280000 \  # 70.000 steps of 4 per batch
                # --fast-dev-run \

# echo time at end
echo "end time: $(date)"
