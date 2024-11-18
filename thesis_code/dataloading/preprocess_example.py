import os
import pathlib

from argparse import ArgumentParser
import nibabel as nib
import torch

from thesis_code.dataloading.transforms import (
    Compose,
    RangeNormalize,
    Resize,
    RemovePercentOutliers,
    Identity,
)
from thesis_code.dataloading.mri_sample import MRISample
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
    parser.add_argument(
        "--preprocess-folder",
        action="store_true",
        help="Preprocess a folder of NIfTI files",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the test suite for this script"
    )
    return parser.parse_args()


def get_transforms(size: int, percent_outliers: float) -> Compose:
    return Compose(
        [
            # Identity(),
            Resize(size),
            RemovePercentOutliers(percent_outliers),
            RangeNormalize(target_min=-1, target_max=1),
        ]
    )


if __name__ == "__main__":
    args = parse_args()
    transforms = get_transforms(args.size, args.percent_outliers)

    # all files ending with .nii.gz in the folder
    if args.preprocess_folder:
        nii_files = list(pathlib.Path(args.nii_path).glob("*.nii.gz"))
    else:
        nii_files = [pathlib.Path(args.nii_path)]
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
            sample = MRISample(image=sample)

            # Apply the transforms
            print("Applying transforms...")
            transformed_sample = transforms(sample)

            # Save the transformed NIfTI file
            print("Saving transformed NIfTI file...")
            transformed_nii = nib.Nifti1Image(  # type: ignore
                transformed_sample["image"].numpy().squeeze(0),
                affine=nii.affine,  # type: ignore
            )

            nib.save(transformed_nii, out_file)  # type: ignore
