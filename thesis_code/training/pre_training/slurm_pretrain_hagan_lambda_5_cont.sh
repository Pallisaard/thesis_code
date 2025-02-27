#!/bin/bash
#SBATCH --job-name=cont_pretrain_lambda_5_640k
#SBATCH --output=slurm_cont_pretrain_lambda_5_640k-%j.out # Name of output file
#SBATCH --error=slurm_cont_pretrain_lambda_5_640k-%j.err # Name of error file
#SBATCH --gres=gpu:a100:1       # Request 4 GPU per job
#SBATCH --time=1-12:00:00    # limit to 36 hours
#SBATCH --cpus-per-task=16  # Number of CPUs for each gpu
#SBATCH --mem=64G        # Memory request
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

python -m thesis_code.training.pre_training.pretrain --model-name "hagan" \
                --latent-dim 1024 \
                --data-path ../data/pre-training/brain-masked-zerosliced \
                --load-from-checkpoint ../checkpoints/pretrained/hagan-l5-1.ckpt \
                --use-all-data-for-training \
                --batch-size 4 \
                --num-workers 14 \
                --accelerator gpu \
                --devices auto \
                --callbacks 'checkpoint' \
                --save-top-k 1 \
                --save-last \
                --log-every-n-steps 50 \
                --max-steps 640000 \
                --lambdas 5.0 \
                # --fast-dev-run \

# echo time at end
echo "end time: $(date)"
