import argparse
import nibabel as nib
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def process_nifti_file(file_path: Path) -> None:
    """
    Process a single NIfTI file by setting values below 1e-4 to 0.0

    Args:
        file_path (Path): Path to the NIfTI file
    """
    try:
        # Load the NIfTI file
        nii = nib.load(str(file_path))

        # Get the data and convert to float32 if not already
        data = nii.get_fdata().astype(np.float32)

        # Apply the threshold operation
        data[data < 1e-3] = 0.0

        # Create a new NIfTI image with the modified data
        new_nii = nib.Nifti1Image(data, nii.affine, nii.header)

        # Save back to the same file
        nib.save(new_nii, str(file_path))
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def process_directory(directory: Path, n_workers: int) -> None:
    """
    Recursively process all .nii.gz files in the given directory and its subdirectories

    Args:
        directory (Path): Path to the directory to process
        n_workers (int): Number of worker processes to use
    """
    # Find all .nii.gz files in the directory and subdirectories
    nifti_files = list(directory.rglob("*.nii.gz"))

    if not nifti_files:
        print(f"No .nii.gz files found in {directory}")
        return

    print(f"Found {len(nifti_files)} .nii.gz files to process")
    print(f"Using {n_workers} worker processes")

    # Process files in parallel
    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_nifti_file, nifti_files),
                total=len(nifti_files),
                desc="Processing files",
            )
        )

    # Report results
    successful = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    print(f"\nProcessing complete! Successfully processed {successful} files")
    if failed > 0:
        print(f"Failed to process {failed} files")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process .nii.gz files by setting values below 1e-4 to 0.0"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing .nii.gz files to process (including subdirectories)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of worker processes to use (default: 8)",
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

    process_directory(directory, args.n_workers)
