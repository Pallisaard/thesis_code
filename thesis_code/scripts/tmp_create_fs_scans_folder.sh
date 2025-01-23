#!/bin/bash
#SBATCH --job-name=tmp_create_fs_scans_folder
#SBATCH --output=slurm_tmp_create_fs_scans_folder_%A_%a.out
#SBATCH --error=slurm_tmp_create_fs_scans_folder_%A_%a.err
#SBATCH --time=1:00:00  # Maximum time for the job
#SBATCH --cpus-per-task=2  # Number of CPUs for each task

module load cuda/11.8
module load cudnn/8.6.0

# If "~/projects/thesis/fastsurfer-output/fine-tuning" does not exist, create it
# mkdir -p ~/projects/thesis/fastsurfer-output/fine-tuning

# if "~/projects/thesis/data/fine-tuning/fs_scans/" does not exist, create it
mkdir -p ~/projects/thesis/data/fine-tuning/fs_scans

source ~/projects/thesis/thesis-code/.venv/bin/activate

# Path to the text file containing subject IDs
SUBJECTS_FILE="~/projects/thesis/data/fine-tuning/nii_gz_files.txt"

# Read each line in the text file
while IFS= read -r line
do
  # Extract the SUBJECT_ID from the line
  SUBJECT_ID=$(basename "$line" .nii.gz)

  # Perform the cp command
  cp ~/projects/thesis/fastsurfer-output/fine-tuning/${SUBJECT_ID}/mri/orig_nu.mgz ~/projects/thesis/data/fine-tuning/fs_scans/${SUBJECT_ID}.mgz

  # Reorient the NIfTI file
  python ~/projects/thesis/thesis-code/thesis_code/scripts/convert_mgz_to_nifti.py ~/projects/thesis/data/fine-tuning/fs_scans/${SUBJECT_ID}.mgz

  # Remove the FastSurfer output
  rm -r ~/projects/thesis/data/fine-tuning/fs_scans/${SUBJECT_ID}.mgz
done < "$SUBJECTS_FILE"