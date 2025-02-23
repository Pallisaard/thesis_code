from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import nibabel as nib
import numpy as np

from thesis_code.scripts.reorient_nii import reorient_nii_to_ras, resample_to_talairach


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
        "--use-splits", action="store_true", help="Use train, test, and val splits."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for parallel processing.",
    )
    return parser.parse_args()


def apply_mask_to_mri(mask_img, original_mri):
    mask_data = mask_img.get_fdata()
    original_mri_data = original_mri.get_fdata()  # type: ignore
    masked_mri_data = original_mri_data * mask_data
    return nib.Nifti1Image(masked_mri_data, original_mri.affine)  # type: ignore


def process_single_file(nii_file, category_dest_dir, fastsurfer_output_dir):
    try:
        # Extract the filename without extension
        nii_file_stem = nii_file.stem.replace(".nii", "")

        # Construct the path to the mask.mgz file
        mask_path: Path = fastsurfer_output_dir / nii_file_stem / "mri" / "mask.mgz"

        # Construct the destination path
        dest_path: Path = category_dest_dir / f"{nii_file_stem}_masked.nii.gz"

        if dest_path.exists():
            return f"Mask already exists: {dest_path}"

        # Check if the mask file exists
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load the mask.mgz file using nibabel
        mask_img = nib.load(mask_path)  # type: ignore
        # Load the original T1w mri
        original_mri = nib.load(nii_file)  # type: ignore

        # Reorient the mask and mri to RAS+
        reoriented_mask = reorient_nii_to_ras(mask_img)
        reoriented_mri = reorient_nii_to_ras(original_mri)

        reoriented_masked_mri = apply_mask_to_mri(reoriented_mask, reoriented_mri)

        # Resample the reoriented masked mri to Talairach space
        resampled_masked_mri: np.ndarray = resample_to_talairach(reoriented_masked_mri)

        # We bound any value below 0.2 to 0.0
        # resampling makes some values slightly below 0.2, which we round to 0.0
        resampled_masked_mri[resampled_masked_mri < 0.2] = 0.0

        # Save the loaded mask to the destination path in .nii.gz format
        nib.save(resampled_masked_mri, str(dest_path))  # type: ignore

        return f"Successfully processed: {nii_file_stem}"
    except Exception as e:
        return f"Error processing {nii_file}: {str(e)}"


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
        for dir_name in ["train", "test", "val"]:
            if not (data_dir / dir_name).exists():
                raise FileNotFoundError(f"{dir_name} directory not found in {data_dir}")
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

        # Create a partial function with fixed arguments
        process_func = partial(
            process_single_file,
            category_dest_dir=category_dest_dir,
            fastsurfer_output_dir=fastsurfer_output_dir,
        )

        # Process files in parallel
        with mp.Pool(processes=args.workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, mask_paths),
                    total=len(mask_paths),
                    desc=f"Processing {category}",
                )
            )

        # Print any errors that occurred during processing
        for result in results:
            if result.startswith("Error"):
                print(result)
