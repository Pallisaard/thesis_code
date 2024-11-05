#!/bin/bash
#SBATCH --job-name=fastsurfer
#SBATCH --output=slurm_fastsurfer_%A_%a.out
#SBATCH --error=slurm_fastsurfer_%A_%a.err
#SBATCH --array=2-879%5  # Array job for 879 MRI files, limit to 5 jobs running at once
#SBATCH --gres=gpu:titanrtx:1       # Request 1 GPU per job
#SBATCH --cpus-per-task=2  # Number of CPUs for each task

module load cuda/11.8
module load cudnn/8.6.0

source ~/venv/bin/activate

# Get the MRI file based on the SLURM array task ID
MRI_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/finetune_dataset/nii_gz_files.txt)

# Extract the subject ID from the file name (or pass another way)
SUBJECT_ID=$(basename $MRI_FILE .nii.gz)

echo "mri file" $MRI_FILE
echo "subject id" $SUBJECT_ID

# Call your existing run_fastsurfer.sh script
bash ~/thesis_code/thesis_code/fastsurfer/run_fastsurfer_finetune_gzj557.sh $MRI_FILE $SUBJECT_ID

# Copy the output to the final directory
cp ~/fastsurfer-output/${SUBJECT_ID}/mri/orig_nu.mgz ~/finetune_dataset/fs_scans/${SUBJECT_ID}.mgz

# Reorient the NIfTI file
python ~/thesis_code/thesis_code/data_collection/reorient_nii.py ~/finetune_dataset/fs_scans/${SUBJECT_ID}.mgz

# Remove the FastSurfer output
rm -r ~/finetune_dataset/fs_scans/${SUBJECT_ID}.mgz