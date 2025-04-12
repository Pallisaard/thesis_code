#!/bin/bash
#SBATCH --job-name=generate_n_sampled_mris
#SBATCH --output=slurm_generate_n_sampled_mris-%j-%a.out # Name of output file
#SBATCH --error=slurm_generate_n_sampled_mris-%j-%a.err # Name of error file
#SBATCH --array=9-9%1
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
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
    9) echo "Vectorizing test dataset"; ;; # test set vectorization
    10) noise=1.0;  clip=1.0;  delta_exp=-5 ;; # no dp sgd
    *) echo "Invalid job ID"; exit 1 ;;
esac

# Convert parameters to checkpoint path format
noise_str=$noise
clip_str=$clip

# Function to run generation for a specific epsilon
run_generation() {
    local epsilon=$1
    local checkpoint_dir
    local output_dir
    
    if [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
        checkpoint_dir="no-dp"
        output_dir="generated-examples-no-dp"
    elif [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
        checkpoint_dir="no-dp-sgd"
        output_dir="generated-examples-no-dp-sgd"
    else
        checkpoint_dir="dp-n${noise_str}-c${clip_str}-s${delta_exp}"
        output_dir="generated-examples-${checkpoint_dir}"
    fi
    
    echo "Running with noise=${noise}, clip=${clip}, delta_exp=${delta_exp}, epsilon=${epsilon}"
    
    python -m thesis_code.evaluation.generate_samples.generate_n_sampled_mris \
        --output-dir "../torch-output/finetune-eval/${output_dir}/epsilon-${epsilon}" \
        --n-samples 250 \
        --use-dp-safe \
        --checkpoint-path "../checkpoints/finetuned/${checkpoint_dir}/epsilon-${epsilon}.pth" \
        --lambdas 5 \
        --device auto \
        --batch-size 2 \
        --vectorizer-dim 2048 \
        --model-name hagan \
        --skip-mri-save \
        --use-custom-checkpoint \
        --custom-checkpoint-map-location cuda || {
        echo "Task n${noise}, c${clip}, s${delta_exp}, epsilon=${epsilon} failed"
        return 1
    }
}

# Run generation for each epsilon value if not task 9
if [ "$SLURM_ARRAY_TASK_ID" -le 8 ]; then
    run_generation 2.00
    run_generation 5.00
    run_generation 10.00
elif [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    run_generation 2.00
    run_generation 5.00
    run_generation 10.00
fi

# Handle test set vectorization if it's task 9
if [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
    echo "Vectorizing test dataset"
    python -m thesis_code.evaluation.generate_samples.vectorize_test_dataset \
        --data-dir "../data/fine-tuning/brain-masked-no-zerosliced" \
        --output-dir "../torch-output/finetune-eval/true-examples-all" \
        --device "cuda" \
        --test-size 250 \
        --make-filename-file \
        --vectorizer-dim 2048 || {
        echo "Test set vectorization failed"
        exit 1
    }
fi
