#!/bin/bash

# Define the datasets
datasets=("ds003653" "ds003114" "ds000140" "ds004856")

# Check if the --flat_join flag is provided
flat_join_flag=""
if [ "$1" == "--flat_join" ]; then
    flat_join_flag="--flat_join"
fi

# Run the Python script with the datasets and optional flat_join flag
python download_and_process_data.py -d "${datasets[@]}" $flat_join_flag