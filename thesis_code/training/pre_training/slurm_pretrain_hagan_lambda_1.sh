#!/bin/bash
#SBATCH --job-name=pretrain_vae_256_lambda_1
#SBATCH --output=slurm_pretrain_hagan-%j.out # Name of output file
#SBATCH --error=slurm_pretrain_hagan-%j.err # Name of error file
#SBATCH --gres=gpu:titanrtx:2       # Request 4 GPU per job
#SBATCH --cpus-per-task=4  # Number of CPUs for each gpu
#SBATCH --mem=16G        # Memory request
#SBATCH --mail-type=END    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=rpa@di.ku.dk # Email

module load cuda/11.8
module load cudnn/8.6.0

source ~/venv/bin/activate

cd ~/thesis_code

# echo devices and nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -m thesis_code.training.pre_training.pretrain --model-name "hagan" \
                --latent-dim 1024 \
                --data-path /home/gzj557/final_dataset \
                --batch-size 4 \
                --num-workers 3 \
                --transforms resize range-normalize remove-percent-outliers \
                --outlier-percentile 0.001 \
                --resize-size 256 \
                --normalize-min -1 \
                --normalize-max 1 \
                --accelerator gpu \
                --strategy ddp_find_unused_parameters_true \
                --devices auto \
                --callbacks checkpoint summary progress \
                --save-top-k 3 \
                --save-last \
                --log-every-n-steps 50 \
                --max-steps 500 \
                --lambda-1 1.0 \
                --lambda-2 1.0 \
                # --fast-dev-run \