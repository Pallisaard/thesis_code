#!/bin/bash
#SBATCH --job-name=finetune_dp_e2 # Name of the job
#SBATCH --output=slurm_finetune_dp_e2-%j.out # Name of output file
#SBATCH --error=slurm_finetune_dp_e2-%j.err # Name of error file
#SBATCH --gres=gpu:a100:1       # Request 4 GPU per job
#SBATCH --time=03:00:00       # Time limit day-hrs:min:sec
#SBATCH --cpus-per-task=10  # Number of CPUs for each gpu
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

# Does not use differential privacy because --use-dp flag is not on.
CUDA_LAUNCH_BLOCKING=1 python -m thesis_code.training.fine_tuning.finetune  --latent-dim 1024 \
                --data-path ../data/fine-tuning/brain-masked \
                --use-dp \
                --use-all-data-for-training \
                --max-epsilon 2.0 \
                --lambdas 5.0 \
                --batch-size 4 \
                --num-workers 8 \
                --device auto \
                --load-from-checkpoint ../checkpoints/pretrained/all-data/hagan_l1_320k.ckpt \
                --val-every-n-steps 1000 \
                --checkpoint-every-n-steps 2500 \
                --checkpoint-path checkpoints/finetuned/dp-epsilon-2 \
                --alphas 1.1 2 3 5 10 20 50 100 \
                --noise-multiplier 1.0 \
                --delta 1e-5 \
                --max-grad-norm 1.0 \

# echo time at end
echo "end time: $(date)"
