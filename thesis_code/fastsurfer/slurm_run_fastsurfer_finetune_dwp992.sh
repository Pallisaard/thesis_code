#!/bin/bash
#SBATCH --job-name=fastsurfer
#SBATCH --output=slurm_fastsurfer_%A_%a.out
#SBATCH --error=slurm_fastsurfer_%A_%a.err
#SBATCH --array=1-409%5  # Array job for 879 MRI files, limit to 5 jobs running at once
#SBATCH --time=10:00  # Maximum time for the job
# #SBATCH --gres=gpu:titanrtx:1       # Request 1 GPU per job
#SBATCH --cpus-per-task=2  # Number of CPUs for each task

module load cuda/11.8
module load cudnn/8.6.0

# If "~/projects/thesis/fastsurfer-output/fine-tuning" does not exist, create it
mkdir -p ~/projects/thesis/fastsurfer-output/fine-tuning

# if "~/projects/thesis/data/fine-tuning/fs_scans/" does not exist, create it
mkdir -p ~/projects/thesis/data/fine-tuning/fs_scans

source ~/projects/thesis/thesis-code/.venv/bin/activate

# Get the MRI file based on the SLURM array task ID
MRI_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/projects/thesis/data/fine-tuning/nii_gz_files.txt)

# Extract the subject ID from the file name (or pass another way)
SUBJECT_ID=$(basename $MRI_FILE .nii.gz)

echo "mri file" $MRI_FILE
echo "subject id" $SUBJECT_ID

# Call your existing run_fastsurfer.sh script
bash ~/projects/thesis/thesis-code/thesis_code/fastsurfer/run_fastsurfer_finetune_dwp992.sh $MRI_FILE $SUBJECT_ID

mv ~/projects/thesis/fastsurfer-output/${SUBJECT_ID} ~/projects/thesis/fastsurfer-output/fine-tuning/.

# Copy the output to the final directory
cp ~/projects/thesis/fastsurfer-output/fine-tuning/${SUBJECT_ID}/mri/orig_nu.mgz ~/projects/thesis/data/fine-tuning/fs_scans/${SUBJECT_ID}.mgz

# Reorient the NIfTI file
python ~/projects/thesis/thesis-code/thesis_code/scripts/convert_mgz_to_nifti.py ~/projects/thesis/data/fine-tuning/fs_scans/${SUBJECT_ID}.mgz

# Remove the FastSurfer output
rm -r ~/projects/thesis/data/fine-tuning/fs_scans/${SUBJECT_ID}.mgz