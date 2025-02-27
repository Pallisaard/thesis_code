#!/bin/bash
#SBATCH --job-name=generate_n_sampled_mris
#SBATCH --output=slurm_generate_n_sampled_mris-%j-%a.out # Name of output file
#SBATCH --error=slurm_generate_n_sampled_mris-%j-%a.err # Name of error file
#SBATCH --array=1-8%1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END    # Mail events (NONE, BEGIN, END, FAIL, ALL)
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
noise_str=$(echo $noise | tr '.' ',')
clip_str=$(echo $clip | tr '.' ',')

# Function to run generation for a specific epsilon
run_generation() {
    local epsilon=$1
    local checkpoint_dir
    local output_dir
    
    if [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
        checkpoint_dir="no-dp"
        output_dir="generated-examples-no-dp"
    else
        checkpoint_dir="generated-examples-dp-n${noise_str}-c${clip_str}-s${delta_exp}"
        output_dir="$checkpoint_dir"
    fi
    
    echo "Running with noise=${noise}, clip=${clip}, delta_exp=${delta_exp}, epsilon=${epsilon}"
    
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris \
        --output-dir "../torch-output/finetune-eval/${output_dir}/epsilon-${epsilon}" \
        --n-samples 1000 \
        --use-dp-safe \
        --checkpoint-path "../checkpoints/finetuned/${checkpoint_dir}/epsilon-${epsilon}.ckpt" \
        --lambdas 5 \
        --device auto \
        --batch-size 2 \
        --from-authors \
        --vectorizer-dim 2048 \
        --model-name hagan \
        --use-custom-checkpoint || {
        echo "Task n${noise}, c${clip}, s${delta_exp}, epsilon=${epsilon} failed"
        return 1
    }
}

# Run generation for each epsilon value
run_generation 2
run_generation 5
run_generation 10
