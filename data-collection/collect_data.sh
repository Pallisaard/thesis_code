#!/bin/bash

# Initialize empty array for datasets
datasets=()
DATA_PATH=""

# Function to print usage
print_usage() {
    echo "Usage: $0 -d|--data-path <path> <dataset1> <dataset2> ..."
    echo "Example: $0 -d /path/to/directory ds004471 ds004392"
    exit 1
}

# Parse command line arguments
# I todally didn't ask ChatGPT to write this. I wrote it myself. I swear.
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-path)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "Error: -d|--data-path requires a directory path"
                print_usage
            fi
            DATA_PATH="$2"
            shift # Remove argument name
            shift # Remove argument value
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            # Add any non-flag argument to the datasets array
            datasets+=("$1")
            shift
            ;;
    esac
done

# Check if data path is provided
if [ -z "$DATA_PATH" ]; then
    echo "Error: Data path is required. Use -d or --data-path to specify the path."
    print_usage
fi

# Check if at least one dataset is provided
if [ ${#datasets[@]} -eq 0 ]; then
    echo "Error: At least one dataset must be specified."
    print_usage
fi

# Print the datasets that will be processed
echo "The following datasets will be processed:"
printf '%s\n' "${datasets[@]}"

# Change to the specified directory
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Directory $DATA_PATH does not exist"
    exit 1
fi

echo "Changing to directory: $DATA_PATH"
cd "$DATA_PATH" || exit 1

# if the exported-datasets directory does not exist, create it
# if it does and --clean-export flag is set, remove all its contents
if [ ! -d "exported-datasets" ]; then
  mkdir exported-datasets
else
  rm -rf exported-datasets/*
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
  datalad get -n "$dataset"
done

# download all T1w nifty images
echo "downloading all T1w nifty images..."
datalad get **/*T1w*.nii.gz

echo "lsing ../"
ls ../

# export all datasets
echo "exporting datasets and moving them to ./.."
for dataset in "${datasets[@]}"; do
  datalad export-archive -d "$dataset" --missing-content continue exported-"$dataset"
  mv exported-"$dataset".tar.gz ../exported-datasets/"$dataset".tar.gz
done

# # cd back to the base directory
echo "changing back to ./.."
cd ..

# # clean up the openneuro directory
echo "cleaning up..."
datalad drop --what all -d openneuro --recursive

# unzip and untar all the tar.gz datasets into a folder with the same name
echo "unzipping and untarring all datasets..."
for dataset in "${datasets[@]}"; do
  tar -xzf exported-datasets/"$dataset".tar.gz -C exported-datasets
done

# remove the tar.gz files
echo "removing tar.gz files..."
rm exported-datasets/*.tar.gz

# make a ../final-dataset/scans folder
if [ ! -d "final-dataset" ]; then
  mkdir final-dataset
  mkdir final-dataset/scans
else
  rm -rf final-dataset/*
  mkdir final-dataset/scans
fi

# move all *T1w*.nii.gz files into it
echo "Moving all T1w NIfTI images to ./final-dataset/scans..."
for dataset in "${datasets[@]}"; do
  mv exported-datasets/"$dataset"/**/*T1w*.nii.gz final-dataset/scans
  for file in final-dataset/scans/*T1w*.nii.gz; do
    mv "$file" final-dataset/scans/"$dataset"-"$(basename "$file")"
  done
done
# mv exported-datasets/**/*T1w*.nii.gz final-dataset/scans
# # Append the dataset name to the filename
# for dataset in "${datasets[@]}"; do
#   for file in final-dataset/scans/*T1w*.nii.gz; do
#     mv "$file" final-dataset/scans/"$dataset"-"$(basename "$file")"
#   done
# done

# for each dataset, create a folder in final-dataset with the same name and move all non-folders from the base dataset (and not its subdirectories) folder into it
echo "Moving all non-T1w NIfTI images to ./final-dataset..."
for dataset in "${datasets[@]}"; do
  mkdir -p final-dataset/"$dataset"
  # find all files in the dataset folder and move them to the final-dataset folder
  find exported-datasets/exported-"$dataset" -maxdepth 1 -type f -exec mv {} final-dataset/"$dataset" \;
done

# remove the exported-datasets directory
echo "removing exported-datasets directory..."
rm -rf exported-datasets

echo "done!"