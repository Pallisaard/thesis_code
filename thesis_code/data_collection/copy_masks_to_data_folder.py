import os
from pathlib import Path
import nibabel as nib
import argparse
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy masks from fastsurfer-output to data folder."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data",
        help="Path to the data directory containing train, test, and val subdirectories.",
    )
    parser.add_argument(
        "--fastsurfer-output-dir",
        type=str,
        default="/fastsurfer-output",
        help="Path to the fastsurfer-output directory containing the masks.",
    )
    return parser.parse_args()


args = parse_args()

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

data_dir = Path(args.data_dir)
fastsurfer_output_dir = Path(args.fastsurfer_output_dir)

# Define source directories
source_dirs = {
    "train": data_dir / "train",
    "test": data_dir / "test",
    "val": data_dir / "val",
}

# Define destination directory
dest_dir = data_dir / "masks"

# Ensure destination directory exists
dest_dir.mkdir(parents=True, exist_ok=True)

# Iterate through each source directory
for category, source_dir in tqdm(source_dirs.items(), desc="Categories"):
    # Create category subdirectory in destination
    category_dest_dir = dest_dir / category
    category_dest_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each file in the source directory
    for nii_file in tqdm(
        source_dir.glob("*.nii.gz"), desc=f"Processing {category}", leave=False
    ):
        # Extract the filename without extension
        file_stem = nii_file.stem

        # Construct the path to the mask.mgz file
        mask_path: Path = fastsurfer_output_dir / file_stem / "mri" / "mask.mgz"

        # Check if the mask file exists
        if not mask_path.exists():
            print(f"Mask file not found: {mask_path}")
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load the mask.mgz file using nibabel
        mask_img = nib.load(mask_path)  # type: ignore

        # Construct the destination path
        dest_path: Path = category_dest_dir / f"{file_stem}_mask.nii.gz"

        # Save the loaded mask to the destination path in .nii.gz format
        nib.save(mask_img, str(dest_path))  # type: ignore

        # print(f"Saved mask to: {dest_path}")
