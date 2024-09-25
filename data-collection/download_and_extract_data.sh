#!/bin/bash

#SBATCH --job-name=download-and-extract-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --mem=16000M

# Function to download and process a dataset
download_and_process() {
    local dataset_name=$1
    local intermediate_name="${dataset_name}_intermediate"
    local output_name="${dataset_name}-extracted-data.tar.gz"

    echo "Processing dataset: ${dataset_name}"

    # Download the dataset
    poetry run python download_dataset.py --data-path ../../data --dataset-id ${dataset_name} --out-data-name ${intermediate_name}.tar.gz

    # Process the downloaded dataset
    poetry run python process_dataset.py --data-path ../../data --input-tar ${intermediate_name}.tar.gz --output-tar ${output_name}

    datalad drop --what all -d ${dataset_name}

    # Clean up intermediate file
    rm ${intermediate_name}.tar.gz

    echo "Completed processing ${dataset_name}"
    echo "Output file: ${output_name}"
    echo
}

# Main script execution
echo "Starting dataset download and processing"

# Process each dataset
download_and_process "ds003653"

echo "All datasets have been downloaded and processed"