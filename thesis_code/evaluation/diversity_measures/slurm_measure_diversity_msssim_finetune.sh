#!/bin/bash
#SBATCH --job-name=measure_msssim_finetune
#SBATCH --output=slurm_measure_msssim_finetune-%j-%a.out
#SBATCH --error=slurm_measure_msssim_finetune-%j-%a.err
#SBATCH --array=9-9%1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=rpa@di.ku.dk

module load cuda/11.8
module load cudnn/8.6.0
module load gcc/13.2.0

cd ~/projects/thesis/thesis-code
source .venv/bin/activate

echo "task id: $SLURM_ARRAY_TASK_ID"
echo

# Calculate parameters based on array task ID
case $SLURM_ARRAY_TASK_ID in
    1) noise=1.0;  clip=1.0;  delta_exp=-3 ;; # baseline s-3
    2) noise=1.0;  clip=1.0;  delta_exp=-5 ;; # baseline s-5
    3) noise=1.0;  clip=1.0;  delta_exp=-7 ;; # baseline s-7
    4) noise=0.75; clip=1.0;  delta_exp=-5 ;; # lower noise
    5) noise=1.5;  clip=1.0;  delta_exp=-5 ;; # higher noise
    6) noise=1.0;  clip=0.75; delta_exp=-5 ;; # lower clip
    7) noise=1.0;  clip=1.5;  delta_exp=-5 ;; # higher clip
    8) noise=1.0;  clip=1.0;  delta_exp=-5 ;; # no dp
    9) noise=1.0;  clip=1.0;  delta_exp=-5 ;; # no dp sgd
    *) echo "Invalid job ID"; exit 1 ;;
esac

# Convert parameters to checkpoint path format
noise_str=$(echo $noise)
clip_str=$(echo $clip)

# Function to measure diversity for a specific epsilon
measure_diversity() {
    local epsilon=$1
    local checkpoint_dir
    local output_dir
    
    if [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
        checkpoint_dir="no-dp"
        output_dir="generated-examples-no-dp"
    elif [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
        checkpoint_dir="no-dp-sgd"
        output_dir="generated-examples-no-dp-sgd"
    else
        checkpoint_dir="dp-n${noise_str}-c${clip_str}-s${delta_exp}"
        output_dir="generated-examples-${checkpoint_dir}"
    fi
    
    echo "Measuring diversity for noise=${noise}, clip=${clip}, delta_exp=${delta_exp}, epsilon=${epsilon}"
    
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --checkpoint-path "../checkpoints/finetuned/${checkpoint_dir}/epsilon-${epsilon}.pth" \
        --device cuda \
        --resolution 256 \
        --output-file "../torch-output/finetune-eval/${output_dir}/epsilon-${epsilon}/msssim_scores.pt" \
        --model-name hagan \
        --n-samples 250 \
        --use-dp-safe \
        --lambdas 5 \
        --use-custom-checkpoint || {
        echo "Task n${noise}, c${clip}, s${delta_exp}, epsilon=${epsilon} failed"
        return 1
    }
}

# \
# --custom-checkpoint-map-location cuda

# Measure diversity for each epsilon value
measure_diversity 2.00
measure_diversity 5.00
measure_diversity 10.00 