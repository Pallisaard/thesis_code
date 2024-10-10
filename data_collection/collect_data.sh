#!/bin/bash

# Enable globstar for recursive globbing on bash
shopt -s globstar

# Initialize array for datasets
datasets=( \
    "ds002790" "ds002785" \
    "ds004711" "ds003653" "ds001747" "ds003826" \
    "ds002345" "ds004285" \
    "ds005026" \
    "ds004217" "ds003849" "ds003717" \
    "ds002242" "ds002655" "ds002898" \
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
echo "Datasets: ${datasets[@]}"

echo "Changing to directory: $DATA_PATH"
cd "$DATA_PATH" || exit 1

# if the exported-datasets directory does not exist, create it
# if it does and --clean-export flag is set, remove all its contents
if [ ! -d "exported-datasets" ]; then
    mkdir exported-datasets
else
    # rm -rf exported-datasets/*
    echo "exported-datasets directory already exists. Please remove it manually if you want to re-export the datasets."
    exit 1
fi

# clone the openneuro dataset
echo "Cloning OpenNeuro datasets..."
datalad clone ///openneuro

# cd to the openneuro directory
echo "changing directory to ./openneuro..."
cd openneuro

# fetch all the datasets
echo "fetching datasets..."
for dataset in "${datasets[@]}"; do
    echo "fetching $dataset..."
    datalad get -n "$dataset"
done

# download all T1w nifty images
echo "downloading all T1w nifty images..."
# datalad get **/*T1w.nii.gz
for dataset in "${datasets[@]}"; do
    echo "downloading T1w's from $dataset..."
    datalad get "$dataset"/sub*/**/*T1w.nii.gz
done

# Special case for ds003097
echo "fetching ds003097..."
datalad get -n ds003097
datalad get ds003097/sub-*/**/*1_T1w.nii.gz
# add ds003097 to the datasets array
datasets+=("ds003097")

# Special case for ds004499
echo "fetching ds004499..."
datalad get -n ds004499
datalad get ds004499/sub-*/**/*1_T1w.nii
# add ds003097 to the datasets array
datasets+=("ds004499")

echo "lsing ../"
ls ../

# export all datasets
echo "exporting datasets and moving them to ./.."
for dataset in "${datasets[@]}"; do
    echo " - exporting $dataset..."
    datalad export-archive -d "$dataset" --missing-content continue exported-"$dataset" 2> /dev/null
    mv exported-"$dataset".tar.gz ../exported-datasets/"$dataset".tar.gz
done

# # cd back to the base directory
echo "changing back to ./.."
cd ..

# # clean up the openneuro directory
# echo "cleaning up..."
# datalad drop --what all -d openneuro --recursive

# unzip and untar all the tar.gz datasets into a folder with the same name
echo "unzipping and untarring all datasets..."
for dataset in "${datasets[@]}"; do
    echo " - unzipping and untarring $dataset..."
    tar -xzf exported-datasets/"$dataset".tar.gz -C exported-datasets
done

# remove the tar.gz files
echo "removing tar.gz files..."
rm exported-datasets/*.tar.gz

# make a ../final_dataset/scans folder
if [ ! -d "final_dataset" ]; then
    mkdir final_dataset
    mkdir final_dataset/scans
else
    echo "final_dataset directory already exists. Removing its contents..."
    exit 1
    # rm -rf final_dataset/*
    # mkdir final_dataset/scans
    
fi

# move all *T1w*.nii.gz files into it
echo "Moving all T1w NIfTI images to ./final_dataset/scans..."
for dataset in "${datasets[@]}"; do
    echo " - Moving all T1w NIfTI images from $dataset to ./final_dataset/scans..."
    # Move files to a temporary directory first
    mkdir -p temp-scans

    if [ "$dataset" = "ds004499" ]; then
        mv exported-datasets/exported-"$dataset"/**/*1_T1w.nii temp-scans
    else
        mv exported-datasets/exported-"$dataset"/**/*T1w*.nii.gz temp-scans
    fi

    # Rename files in the temporary directory
    for file in temp-scans/*T1w*.nii.gz; do
        mv "$file" final_dataset/scans/"$dataset"-"$(basename "$file")"
    done

    # Clean up the temporary directory
    rm -rf temp-scans
done

# for each dataset, create a folder in final_dataset with the same name and move all non-folders from the base dataset (and not its subdirectories) folder into it
echo "Moving all non-T1w NIfTI images to ./final_dataset..."
for dataset in "${datasets[@]}"; do
    mkdir -p final_dataset/"$dataset"
    # find all files in the dataset folder and move them to the final_dataset folder
    find exported-datasets/exported-"$dataset" -maxdepth 1 -type f -exec mv {} final_dataset/"$dataset" \;
done

# remove the exported-datasets directory
echo "removing exported-datasets directory..."
rm -rf exported-datasets

echo "done!"