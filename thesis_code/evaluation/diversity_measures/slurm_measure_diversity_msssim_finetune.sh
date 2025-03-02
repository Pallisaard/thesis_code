#!/bin/bash
#SBATCH --job-name=measure_msssim_finetune
#SBATCH --output=slurm_measure_msssim_finetune-%j-%a.out
#SBATCH --error=slurm_measure_msssim_finetune-%j-%a.err
#SBATCH --array=1-8%1
#SBATCH --gres=gpu:a100:1
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
    *) echo "Invalid job ID"; exit 1 ;;
esac

# Convert parameters to checkpoint path format
noise_str=$(echo $noise)
clip_str=$(echo $clip)

# Function to measure diversity for a specific epsilon
measure_diversity() {
    local epsilon=$1
    local input_dir
    
    if [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
        input_dir="generated-examples-no-dp"
    else
        input_dir="generated-examples-dp-n${noise_str}-c${clip_str}-s${delta_exp}"
    fi
    
    echo "Measuring diversity for noise=${noise}, clip=${clip}, delta_exp=${delta_exp}, epsilon=${epsilon}"
    
    python -m thesis_code.evaluation.diversity_measures.measure_diversity_msssim \
        --input-dir "../torch-output/finetune-eval/${input_dir}/epsilon-${epsilon}" \
        --device cuda \
        --resolution 256 \
        --output-file "../torch-output/finetune-eval/${input_dir}/epsilon-${epsilon}/msssim_scores.pt" || {
        echo "Task n${noise}, c${clip}, s${delta_exp}, epsilon=${epsilon} failed"
        return 1
    }
}

# Measure diversity for each epsilon value
measure_diversity 2
measure_diversity 5
measure_diversity 10 