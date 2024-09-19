import argparse
import os
import tarfile
import tempfile
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and reorganize exported dataset."
    )
    parser.add_argument(
        "--data-path",
        default=".",
        help="Path to the data directory (default: current directory)",
    )
    parser.add_argument(
        "--input-tar", required=True, help="Name of the input tar.gz file"
    )
    parser.add_argument(
        "--output-tar", required=True, help="Name of the output tar.gz file"
    )
    return parser.parse_args()


def process_dataset(data_path, input_tar, output_tar):
    input_tar_path = Path(data_path) / input_tar
    output_tar_path = Path(data_path) / output_tar

    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the input tar.gz file
        with tarfile.open(input_tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Find the extracted directory (assuming it's the only directory in temp_dir)
        extracted_dir = next(Path(temp_dir).iterdir())

        # Create a new directory structure
        output_dir = Path(temp_dir) / "processed_dataset"
        output_dir.mkdir()
        scans_dir = output_dir / "scans"
        scans_dir.mkdir()

        # Copy files from the root directory
        for item in extracted_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, output_dir)

        # Find and copy all .nii.gz files to the scans directory
        for nii_file in extracted_dir.rglob("*.nii.gz"):
            shutil.copy2(nii_file, scans_dir)

        # Create the new tar.gz file
        with tarfile.open(output_tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=".")


def main():
    args = parse_args()
    process_dataset(args.data_path, args.input_tar, args.output_tar)
    print(f"Processed dataset saved to {os.path.join(args.data_path, args.output_tar)}")


if __name__ == "__main__":
    main()
