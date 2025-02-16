import argparse
import nibabel as nib
from pathlib import Path
import numpy as np


def process_nifti_file(file_path: Path) -> None:
    """
    Process a single NIfTI file by setting values below 1e-4 to 0.0

    Args:
        file_path (Path): Path to the NIfTI file
    """
    print(f"Processing {file_path}")

    # Load the NIfTI file
    nii = nib.load(str(file_path))

    # Get the data and convert to float32 if not already
    data = nii.get_fdata().astype(np.float32)

    # Apply the threshold operation
    data[data < 1e-4] = 0.0

    # Create a new NIfTI image with the modified data
    new_nii = nib.Nifti1Image(data, nii.affine, nii.header)

    # Save back to the same file
    nib.save(new_nii, str(file_path))


def process_directory(directory: Path) -> None:
    """
    Recursively process all .nii.gz files in the given directory and its subdirectories

    Args:
        directory (Path): Path to the directory to process
    """
    # Find all .nii.gz files in the directory and subdirectories
    nifti_files = list(directory.rglob("*.nii.gz"))

    if not nifti_files:
        print(f"No .nii.gz files found in {directory}")
        return

    print(f"Found {len(nifti_files)} .nii.gz files to process")

    # Process each file
    for file_path in nifti_files:
        process_nifti_file(file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process .nii.gz files by setting values below 1e-4 to 0.0"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing .nii.gz files to process (including subdirectories)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    directory = Path(args.directory)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        exit(1)

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        exit(1)

    process_directory(directory)
    print("Processing complete!")
