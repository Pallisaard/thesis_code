import os
from pathlib import Path

from argparse import ArgumentParser
import nibabel as nib
import numpy as np
import torch

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
    return parser.parse_args()


def get_transforms(
    size: int, percent_outliers: float, remove_sero_slices: bool
) -> Compose:
    return Compose(
        [
            RemoveZeroZSlices() if remove_sero_slices else Identity(),
            Resize(size),
            RemovePercentOutliers(percent_outliers),
            RangeNormalize(target_min=-1, target_max=1),
        ]
    )


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
    file_names = [f.name for f in nii_files]

    for nii_path, nii_name in tqdm(zip(nii_files, file_names), total=len(nii_files)):
        print(f"Processing {nii_name}...")
        out_file = os.path.join(args.out_path, nii_name)

        if args.test:
            print(f"Test mode: not saving to {out_file}")

        else:
            # Load the NIfTI file
            nii = nib.load(nii_path)  # type: ignore
            sample = torch.from_numpy(nii.get_fdata()).unsqueeze(0)  # type: ignore

            zoom_factors = (
                args.size / sample.shape[1],
                args.size / sample.shape[2],
                args.size / sample.shape[3],
            )
            scale_matrix = np.diag(
                [zoom_factors[0], zoom_factors[1], zoom_factors[2]],
                1,
            )
            new_affine = nii.affine @ scale_matrix

            # Apply the transforms
            print("Applying transforms...")
            transformed_sample = transforms(sample)

            # Save the transformed NIfTI file
            print("Saving transformed NIfTI file...")
            transformed_nii = nib.Nifti1Image(  # type: ignore
                transformed_sample.numpy().squeeze(0),
                affine=new_affine,  # type: ignore
            )

            nib.save(transformed_nii, out_file)  # type: ignore
