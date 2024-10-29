#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1
output_file="transformed_files.txt"

# Create the output file if it doesn't exist
touch $output_file

# Read the list of already transformed files into an array
transformed_files=()
while IFS= read -r line; do
    transformed_files+=("$line")
done < $output_file

for file in "$directory"/*.nii.gz; do
    if [ -f "$file" ]; then
        # Skip the file if it is already transformed
        if [[ " ${transformed_files[*]} " =~ ${file} ]]; then
            echo "Skipping already transformed file: $file"
            continue
        fi

        reoriented_file="${file%.nii.gz}_re.nii.gz"

        # Run fslreorient2std
        fslreorient2std "$file" "$reoriented_file"

        # Remove the original file
        rm "$file"

        # Rename the reoriented file to the original file name
        mv "$reoriented_file" "$file"

        # Append the processed file name to the output file
        echo "$file" >> $output_file
    fi
done

echo "Processing complete. Transformed files are listed in $output_file."