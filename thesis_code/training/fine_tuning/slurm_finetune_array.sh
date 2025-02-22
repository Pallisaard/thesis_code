#!/bin/bash
#SBATCH --job-name=finetune_dp_array
#SBATCH --output=slurm_finetune_dp_array-%A_%a.out
#SBATCH --error=slurm_finetune_dp_array-%A_%a.err
#SBATCH --array=8-8%1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=rpa@di.ku.dk


module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

# Calculate parameters based on array task ID
# We have 8 combinations total:
# n1.0-c1.0-s-3, n1.0-c1.0-s-5, n1.0-c1.0-s-7  (baseline with different deltas)
# n0.75-c1.0-s-5, n1.5-c1.0-s-5                 (noise variations)
# n1.0-c0.75-s-5, n1.0-c1.5-s-5                 (clip variations)
# no-dp                                          (baseline without DP)

case $SLURM_ARRAY_TASK_ID in
    1) noise=1.0;  clip=1.0;  delta_exp=-3; use_dp=true  ;; # baseline s-3
    2) noise=1.0;  clip=1.0;  delta_exp=-5; use_dp=true  ;; # baseline s-5
    3) noise=1.0;  clip=1.0;  delta_exp=-7; use_dp=true  ;; # baseline s-7
    4) noise=0.75; clip=1.0;  delta_exp=-5; use_dp=true  ;; # lower noise
    5) noise=1.5;  clip=1.0;  delta_exp=-5; use_dp=true  ;; # higher noise
    6) noise=1.0;  clip=0.75; delta_exp=-5; use_dp=true  ;; # lower clip
    7) noise=1.0;  clip=1.5;  delta_exp=-5; use_dp=true  ;; # higher clip
    8) noise=1.0;  clip=1.0;  delta_exp=-5; use_dp=false ;; # no dp
    *) echo "Invalid job ID"; exit 1 ;;
esac

if [ "$use_dp" = true ]; then
    # Convert parameters to checkpoint path format for DP runs
    noise_str=$(echo $noise | tr '.' ',')
    clip_str=$(echo $clip | tr '.' ',')
    delta=$(echo "10^$delta_exp" | bc -l)
    checkpoint_dir="checkpoints/finetuned/dp-n${noise_str}-c${clip_str}-s${delta_exp}"
    
    echo "Running DP experiment with:"
    echo "Noise multiplier: $noise"
    echo "Clip norm: $clip"
    echo "Delta: $delta"
else
    checkpoint_dir="checkpoints/finetuned/no-dp"
    echo "Running non-DP experiment"
fi

echo "Checkpoint dir: $checkpoint_dir"
echo "Start time: $(date)"

# Base command with common parameters
cmd="python -m thesis_code.training.fine_tuning.finetune \
    --latent-dim 1024 \
    --data-path ../data/fine-tuning/brain-masked-no-zerosliced \
    --use-all-data-for-training \
    --max-epsilons 2.0 5.0 10.0 \
    --lambdas 5.0 \
    --batch-size 4 \
    --num-workers 14 \
    --device auto \
    --load-from-checkpoint ../checkpoints/pretrained/hagan-l5-1.ckpt \
    --val-every-n-steps 1000 \
    --checkpoint-every-n-steps 2500 \
    --checkpoint-path \"$checkpoint_dir\" \
    --alphas 1.1 2 3 5 10 20 50 100 200 500 1000"

if [ "$use_dp" = true ]; then
    # Add DP-specific parameters
    cmd="$cmd \
    --use-dp \
    --noise-multiplier $noise \
    --delta $delta \
    --max-grad-norm $clip"
fi

# Execute the command
eval $cmd

echo "End time: $(date)"
