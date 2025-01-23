import argparse
import nibabel as nib
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a .mgz file to .nii.gz format.\n\n"
        "This script takes a .mgz file as input and saves it as a .nii.gz file.\n\n"
        "Example usage:\n"
        "  python reorient_nii.py input_file.mgz\n"
        "This will produce a file named input_file.nii.gz in the same directory."
    )
    parser.add_argument("input_path", type=str, help="Path to the input .mgz file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # For a single .mgz file
    mgz_path = Path(args.input_path)
    mgz = nib.load(mgz_path)  # type: ignore

    # Replace the extension of the input file with .nii.gz
    output_path = args.input_path.replace(".mgz", ".nii.gz")
    nib.save(mgz, output_path)  # type: ignore
