#!/bin/bash

DATA_PATH="$1"

# Check if DATA_PATH is set
if [ -z "$DATA_PATH" ]; then
  echo "Error: Data path is required"
  echo "Usage: $0 <data-path>"
  echo "Example: $0 /path/to/directory"
  exit 1
fi

echo "Data path: $DATA_PATH"

python thesis_code/scripts/convert_dicom_2_nifti.py $DATA_PATH/zip_files $DATA_PATH/nifti_files
python thesis_code/scripts/filter_unique_finetune_niftis.py --input-dir $DATA_PATH/nifti_files --output-dir $DATA_PATH/unique_nifti_files
python thesis_code/scripts/find_all_nii_files.py $DATA_PATH/unique_niftis --replace $DATA_PATH/unique_nifti_files:/data/
