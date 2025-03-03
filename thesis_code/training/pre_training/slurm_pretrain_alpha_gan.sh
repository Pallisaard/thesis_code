#!/bin/bash
#SBATCH --job-name=pretrain_alpha_gan
#SBATCH --output=slurm_pretrain_alpha_gan-%j.out # Name of output file
#SBATCH --error=slurm_pretrain_alpha_gan-%j.err # Name of error file
#SBATCH --gres=gpu:l40s:1       # Request 4 GPU per job
#SBATCH --cpus-per-task=8  # Number of CPUs for each gpu
#SBATCH --time=24:00:00    # Limit to 36 hours.
#SBATCH --mem=32G        # Memory request
#SBATCH --mail-type=END    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email
#SBATCH --dependency=afterany:5109_1

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate


# echo devices and nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# echo time at start
echo "start time: $(date)"

python -m thesis_code.training.pre_training.pretrain --model-name "alpha_gan" \
                --latent-dim 1024 \
                --data-path ../data/pre-training/brain-masked-no-zerosliced-64 \
                --use-all-data-for-training \
                --batch-size 16 \
                --num-workers 6 \
                --accelerator gpu \
                --devices auto \
                --callbacks 'checkpoint' \
                --save-top-k 3 \
                --save-last \
                --log-every-n-steps 50 \
                --max-steps 200000 \
                --lambdas 10.0 \
                # --fast-dev-run \

# echo time at end
echo "end time: $(date)"