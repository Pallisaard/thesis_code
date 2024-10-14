#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1
processed_file="already_processed.txt"


# change directory to the specified directory
echo "Changing to directory: $directory"
cd "$directory" || exit 1

# Check if the output file exists
if [ ! -f "$processed_file" ]; then
    echo "Output file $processed_file not found."
    exit 1
fi

# Read the output file into a list
file_list=()
while IFS= read -r line; do
    file_list+=("$line")
done < "$processed_file"

# Process each *.nii.gz file in the current directory
for file in *.nii.gz; do
    echo "Processing $file..."
    if [[ ! " ${file_list[*]} " =~  ${file}  ]]; then
        reoriented_file="${file%.nii.gz}_re.nii.gz"

        # Run fslreorient2std
        echo "- Reorienting $file..."
        fslreorient2std "$file" "$reoriented_file"

        # Remove the original file
        echo "- Removing $file..."
        rm -f "$file"

        # Rename the reoriented file to the original file name
        echo "- Renaming $reoriented_file to $file..."
        mv -f "$reoriented_file" "$file"

        # Append the processed file name to the processed file
        echo "- logging $file as transformed.."
        echo "$file" >> $processed_file
    fi
done

# Rename the output file to already_processed

echo "Processing complete. Processed files are listed in $processed_file."