import subprocess
import os
import argparse
import tarfile
import shutil


def main():
    parser = argparse.ArgumentParser(description="Download and process datasets.")
    parser.add_argument(
        "-d",
        "--dataset_names",
        nargs="+",
        required=True,
        help="List of dataset names to download and process.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=".",
        help="Path to the data directory (default: current directory)",
    )
    parser.add_argument(
        "--flat-join",
        action="store_true",
        help="Boolean flag for flat join (functionality to be implemented).",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="extracted_data",
        help="Default name for the output folder name. (default: extracted_data)",
    )

    args = parser.parse_args()

    print("Starting dataset download and processing")

    # Process each dataset
    for dataset in args.dataset_names:
        download_and_process(dataset, args.out_name, args.flat_join)

    print("All datasets have been downloaded and processed")


def download_and_process(dataset_name, out_name, flat_join):
    intermediate_name = f"{dataset_name}-intermediate"
    output_name_long = f"{dataset_name}-extracted.tar.gz"
    output_name_short = f"{dataset_name}.tar.gz"
    extracted_folder = out_name

    print(f"Processing dataset: {dataset_name}")

    # Download the dataset
    subprocess.run(
        [
            "python",
            "download_dataset.py",
            "--dataset-name",
            dataset_name,
            "--out-data-name",
            f"{intermediate_name}.tar.gz",
        ],
        check=True,
    )

    # Process the downloaded dataset
    subprocess.run(
        [
            "python",
            "process_dataset.py",
            "--input-tar",
            f"{intermediate_name}.tar.gz",
            "--output-tar",
            output_name_long,
            "--flat_join" if flat_join else "",
        ],
        check=True,
    )

    # Clean up intermediate file
    os.remove(f"{intermediate_name}.tar.gz")

    # Remove the extracted_data folder if it exists
    # Ensure the extracted_data folder exists and is empty
    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)
    os.makedirs(extracted_folder)

    # Move the extracted tar.gz to the extracted_data folder
    shutil.move(output_name_long, os.path.join(extracted_folder, output_name_short))

    # Unpack the tar.gz file and rename the folder
    with tarfile.open(os.path.join(extracted_folder, output_name_short), "r:gz") as tar:
        tar.extractall(path=extracted_folder)

    # Remove the tar.gz file after extraction
    os.remove(os.path.join(extracted_folder, output_name_short))

    print(f"Completed processing {dataset_name}")
    print(f"Output file: {output_name_short}")
    print()


if __name__ == "__main__":
    main()
