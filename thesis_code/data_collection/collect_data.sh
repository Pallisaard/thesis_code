#!/bin/bash

# Enable globstar for recursive globbing on bash
shopt -s globstar

# Initialize array for datasets
datasets=(
  "ds002790" "ds002785"
  "ds004711" "ds003653" "ds001747" "ds003826"
  "ds002345" "ds004285"
  "ds005026"
  "ds004217" "ds003849" "ds003717"
  "ds002242" "ds002655"
  "ds003097" "ds004499" "ds003768"
) # Add your datasets here
DATA_PATH="$1"

# Check if DATA_PATH is set
if [ -z "$DATA_PATH" ]; then
  echo "Error: Data path is required"
  echo "Usage: $0 <data-path>"
  echo "Example: $0 /path/to/directory"
  exit 1
fi

# Your logic to handle datasets and DATA_PATH goes here
echo "Data path: $DATA_PATH"
echo "Datasets: ${datasets[*]}"

echo "Changing to directory: $DATA_PATH"
cd "$DATA_PATH" || exit 1

# if the exported_datasets directory does not exist, create it
# if it does and --clean-export flag is set, remove all its contents
if [ ! -d "exported_datasets" ]; then
  mkdir exported_datasets
else
  echo "exported_datasets directory already exists. Please remove it manually if you want to re-export the datasets."
  exit 1
fi

# clone the openneuro dataset
echo "Cloning OpenNeuro datasets..."
datalad clone ///openneuro

# cd to the openneuro directory
echo "changing directory to ./openneuro..."
cd openneuro || exit 1

# Check if exported_datasets.txt exists, create it if not
if [ ! -f "../exported_datasets.txt" ]; then
  touch ../exported_datasets.txt
fi

# Read exported datasets into an array
exported_datasets=()
while IFS= read -r line; do
  exported_datasets+=("$line")
done <../exported_datasets.txt

# Process all datasets
ls
echo "Processing datasets..."
for dataset in "${datasets[@]}"; do
  echo "Processing $dataset..."

  # Fetch dataset if not already exported
  if [[ ! " ${exported_datasets[*]} " =~ ${dataset} ]]; then
    echo "  - fetching $dataset..."
    datalad get -n "$dataset"

    echo "  - downloading T1w's from $dataset..."
    case "$dataset" in
    "ds003097") datalad get "$dataset"/sub*/**/*1_T1w.nii.gz ;;
    "ds004499") datalad get "$dataset"/sub*/**/*T1w.nii ;;
    *) datalad get "$dataset"/sub*/**/*T1w.nii.gz ;;
    esac

    echo "  - exporting $dataset..."
    datalad export-archive -d "$dataset" --missing-content continue exported_"$dataset" 2>&1 | tee -a collect_data_errs.log
    mv exported_"$dataset".tar.gz ../exported_datasets/exported_"$dataset".tar.gz
  else
    echo "  - Skipping $dataset as it is already exported."
  fi
done

# cd back to the base directory
echo "changing back to ./.."
cd ..
ls

# unzip and untar all the tar.gz datasets into a folder with the same name
echo "unzipping and untarring all datasets..."
for dataset in "${datasets[@]}"; do
  if [[ ! " ${exported_datasets[*]} " =~ ${dataset} ]]; then
    echo " - unzipping and untarring $dataset..."
    tar -xzf exported_datasets/exported_"$dataset".tar.gz -C exported_datasets
  fi
done

# make a ../final_dataset/scans folder
if [ ! -d "final_dataset" ]; then
  mkdir final_dataset
  mkdir final_dataset/scans
fi

# move all *T1w*.nii.gz files into it
echo "Moving all T1w NIfTI images to ./final_dataset/scans..."
for dataset in "${datasets[@]}"; do
  if [[ ! " ${exported_datasets[*]} " =~ ${dataset} ]]; then
    echo " - Moving all T1w NIfTI images from $dataset to ./final_dataset/scans..."
    mkdir -p temp_scans

    if [ "$dataset" = "ds004499" ]; then
      # Compress all files in exported_datasets to .nii.gz
      echo "Compressing all files in exported_datasets/exported_$dataset to .nii.gz..."
      gzip exported_datasets/exported_"$dataset"/**/*T1w.nii
    fi
    mv -f exported_datasets/exported_"$dataset"/**/*T1w*.nii.gz temp_scans

    for file in temp_scans/*T1w*.nii.gz; do
      mv -f "$file" final_dataset/scans/"$dataset"_"$(basename "$file")"
    done

    rm -rf temp_scans
  fi
done

# for each dataset, create a folder in final_dataset with the same name and move all non-folders from the base dataset (and not its subdirectories) folder into it
echo "Moving all non-T1w NIfTI images to ./final_dataset..."
for dataset in "${datasets[@]}"; do
  if [[ ! " ${exported_datasets[*]} " =~ ${dataset} ]]; then
    mkdir -p final_dataset/"$dataset"
    find exported_datasets/exported_"$dataset" -maxdepth 1 -type f -exec mv -f {} final_dataset/"$dataset" \;
  fi
done

# remove the exported_datasets directory
echo "removing exported_datasets directory..."
rm -rf exported_datasets

for dataset in "${datasets[@]}"; do
  if [[ ! " ${exported_datasets[*]} " =~ ${dataset} ]]; then
    echo "$dataset" >>exported_datasets.txt
  fi
done

# Misc: Arbitrary cleanup
# This scan is broken for some reason (i swear to god fuck)
rm -f final_dataset/scans/ds004217_sub-011_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-014_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-018_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-064_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-070_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-071_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-074_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-077_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-082_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-083_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-084_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-085_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-090_T1w.nii.gz
rm -f final_dataset/scans/ds004217_sub-025_T1w.nii.gz
echo "done!"
