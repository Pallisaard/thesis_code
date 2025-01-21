import os
from pathlib import Path
from typing import Optional
import nibabel as nib
import argparse
from tqdm import tqdm

from thesis_code.scripts.reorient_nii import reorient_nii_to_ras
from thesis_code.dataloading.transforms import (
    MRITransform,
    Compose,
    RangeNormalize,
    Resize,
    RemovePercentOutliers,
    RemoveZeroSlices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply masks from fastsurfer-output to NIfTI files in data folder."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to the data directory containing train, test, and val subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--fastsurfer-output-dir",
        type=str,
        help="Path to the fastsurfer-output directory containing the masks.",
    )
    parser.add_argument(
        "--use_splits", action="store_true", help="Use train, test, and val splits."
    )
    # parser.add_argument("--size", type=int, default=256)
    # parser.add_argument("--percent-outliers", type=float, default=0.001)
    return parser.parse_args()


def apply_mask_to_mri(mask_img, original_mri, transforms: Optional[MRITransform]):
    reoriented_mask_img = reorient_nii_to_ras(mask_img)
    reoriented_mask_data = reoriented_mask_img.get_fdata()
    original_mri_data = original_mri.get_fdata()  # type: ignore
    masked_mri_data = original_mri_data * reoriented_mask_data
    if transforms is not None:
        masked_mri_data = transforms.transform_np(masked_mri_data)
    return nib.Nifti1Image(masked_mri_data, original_mri.affine)  # type: ignore


if __name__ == "__main__":
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    else:
        print(f"Data directory found: {data_dir}")

    fastsurfer_output_dir = Path(args.fastsurfer_output_dir)
    if not fastsurfer_output_dir.exists():
        raise FileNotFoundError(
            f"FastSurfer output directory not found: {fastsurfer_output_dir}"
        )
    else:
        print(f"FastSurfer output directory found: {fastsurfer_output_dir}")

    # Define source directories
    if args.use_splits:
        source_dirs = {
            "train": data_dir / "train",
            "test": data_dir / "test",
            "val": data_dir / "val",
        }
    else:
        source_dirs = {"all": data_dir}

    # Define destination directory
    dest_dir = Path(args.output_dir)

    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each source directory
    for category, source_dir in tqdm(source_dirs.items(), desc="Categories"):
        print("Processing category:", category)
        # Create category subdirectory in destination
        category_dest_dir = dest_dir / category
        category_dest_dir.mkdir(parents=True, exist_ok=True)

        # Get generator of all .nii files in the source directory
        mask_paths = list(source_dir.glob("*.nii.gz"))

        # Iterate through each file in the source directory
        for nii_file in tqdm(mask_paths, desc=f"Processing {category}", leave=False):
            # Extract the filename without extension
            nii_file_stem = nii_file.stem.replace(".nii", "")

            # Construct the path to the mask.mgz file
            mask_path: Path = fastsurfer_output_dir / nii_file_stem / "mri" / "mask.mgz"

            # Check if the mask file exists
            if not mask_path.exists():
                print(f"Mask file not found: {mask_path}")
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

            # Construct the destination path
            dest_path: Path = category_dest_dir / f"{nii_file_stem}_masked.nii.gz"

            if dest_path.exists():
                print(f"Mask already exists: {dest_path}")
                continue

            # Load the mask.mgz file using nibabel
            mask_img = nib.load(mask_path)  # type: ignore
            # Load the original T1w mri
            original_mri = nib.load(nii_file)  # type: ignore

            reoriented_mask_img = reorient_nii_to_ras(mask_img)
            masked_mri = apply_mask_to_mri(
                reoriented_mask_img, original_mri, transforms=None
            )

            # Save the loaded mask to the destination path in .nii.gz format
            nib.save(masked_mri, str(dest_path))  # type: ignore

            # print(f"Saved mask to: {dest_path}")
