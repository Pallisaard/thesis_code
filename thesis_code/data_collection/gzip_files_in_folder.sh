#!/bin/bash

# Check if a directory is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Check if the provided argument is a directory
if [ ! -d "$1" ]; then
  echo "Error: '$1' is not a directory."
  exit 1
fi

# Change to the specified directory
cd "$1"

# Loop through all files in the directory, gzip them, and remove the originals
for f in *; do
  gzip "$f" && rm "$f"
done

echo "All files in '$1' have been gzipped and the originals removed."