#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1
output_file="processed.txt"

# Clear the output file
> $output_file

for file in "$directory"/*.nii.gz; do
    if [ -f "$file" ]; then
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

echo "Processing complete. Processed files are listed in $output_file."