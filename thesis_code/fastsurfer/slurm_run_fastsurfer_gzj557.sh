#!/bin/bash
#SBATCH --job-name=fastsurfer
#SBATCH --output=fastsurfer_%A_%a.out
#SBATCH --error=fastsurfer_%A_%a.err
#SBATCH --array=1-2740%5   # Array job for 2740 MRI files, limit to 5 jobs running at once
#SBATCH --gres=gpu:1       # Request 1 GPU per job
#SBATCH --cpus-per-task=4  # Number of CPUs for each task



# If /data/final_dataset/scans/ does not exist, throw error
if [ ! -d /data/final_dataset/scans/ ] ; then
    echo "Directory /data/final_dataset/scans/ does not exist."
    exit 1
fi

# Get the MRI file based on the SLURM array task ID
MRI_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /data/final_dataset/mri_file_list.txt)

# Extract the subject ID from the file name (or pass another way)
SUBJECT_ID=$(basename $MRI_FILE .nii.gz)

# Call your existing run_fastsurfer.sh script
./run_fastsurfer_gzj557.sh $MRI_FILE $SUBJECT_ID

# Copy the output to the final directory
cp ~/fastsurfer-output/${SUBJECT_ID}/mri/orig_nu.mgz ~/final_dataset/fs_scans/${SUBJECT_ID}.mgz

# Reorient the NIfTI file
python ~/thesis_code/thesis_code/data_collection/reorient_nii.py ~/final_dataset/fs_scans/${SUBJECT_ID}.mgz