import os
from pathlib import Path

from argparse import ArgumentParser
import nibabel as nib
import numpy as np
import torch
from multiprocessing import Pool

from thesis_code.dataloading.transforms import (
    Compose,
    Identity,
    RangeNormalize,
    Resize,
    RemovePercentOutliers,
    RemoveZeroZSlices,
)
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--nii-path", type=str, required=True, help="Path to the NIfTI file"
    )
    parser.add_argument(
        "--out-path", type=str, help="Path to save the transformed NIfTI file"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--percent-outliers", type=float, default=0.001)
    parser.add_argument("--remove-zero-slices", action="store_true")
    parser.add_argument(
        "--preprocess-folder",
        action="store_true",
        help="Preprocess a folder of NIfTI files",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the test suite for this script"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for preprocessing",
    )
    return parser.parse_args()


def get_transforms(
    size: int, percent_outliers: float, remove_zero_slices: bool
) -> Compose:
    return Compose(
        [
            RemoveZeroZSlices() if remove_zero_slices else Identity(),
            Resize(size),
            RemovePercentOutliers(percent_outliers),
            RangeNormalize(target_min=-1, target_max=1),
        ]
    )


def process_single_file(args_tuple):
    nii_path, out_path, transforms, test = args_tuple
    out_file = os.path.join(out_path, nii_path.name)

    if test:
        return

    # Load the NIfTI file
    nii = nib.load(nii_path)
    sample = torch.from_numpy(nii.get_fdata()).unsqueeze(0)

    zoom_factors = (
        256 / sample.shape[1],
        256 / sample.shape[2],
        256 / sample.shape[3],
    )
    # Create a 4x4 identity matrix
    scale_matrix = np.eye(4)
    # Set the scaling factors in the first 3 diagonal elements
    scale_matrix[0:3, 0:3] = np.diag(
        [zoom_factors[0], zoom_factors[1], zoom_factors[2]]
    )
    new_affine = nii.affine @ scale_matrix

    # Apply the transforms
    transformed_sample = transforms(sample)

    # Save the transformed NIfTI file
    transformed_nii = nib.Nifti1Image(
        transformed_sample.numpy().squeeze(0),
        affine=new_affine,
    )

    nib.save(transformed_nii, out_file)
    return nii_path.name


if __name__ == "__main__":
    args = parse_args()
    transforms = get_transforms(
        args.size, args.percent_outliers, args.remove_zero_slices
    )

    if not Path(args.out_path).exists():
        raise FileNotFoundError(f"Output path not found: {args.out_path}")
    else:
        print(f"Saving transformed NIfTI files to {args.out_path}")

    if not Path(args.nii_path).exists():
        raise FileNotFoundError(f"NIfTI file not found: {args.nii_path}")
    else:
        print(f"Preprocessing NIfTI files from {args.nii_path}")

    # all files ending with .nii.gz in the folder
    if args.preprocess_folder:
        nii_files = list(Path(args.nii_path).glob("*.nii.gz"))
    else:
        nii_files = [Path(args.nii_path)]

    # Prepare arguments for parallel processing
    process_args = [
        (nii_path, args.out_path, transforms, args.test) for nii_path in nii_files
    ]

    # Create a pool of workers and process files in parallel
    print(f"Processing {len(nii_files)} files using {args.n_workers} workers...")
    with Pool(processes=args.n_workers) as pool:
        for filename in tqdm(
            pool.imap(process_single_file, process_args),
            total=len(process_args),
            desc="Processing files",
        ):
            if filename:  # Will be None for test mode
                tqdm.write(f"Completed processing: {filename}")
