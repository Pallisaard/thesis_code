from thesis_code.dataloading.transforms import (
    Compose,
    RangeNormalize,
    Resize,
    RemovePercentOutliers,
)
from argparse import ArgumentParser
from thesis_code.dataloading.mri_sample import MRISample
import nibabel as nib
import torch


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
    return parser.parse_args()


def get_transforms(size: int, percent_outliers: float) -> Compose:
    return Compose(
        [
            Resize(size),
            RemovePercentOutliers(percent_outliers),
            RangeNormalize(target_min=-1, target_max=1),
        ]
    )


if __name__ == "__main__":
    args = parse_args()
    transforms = get_transforms(args.size, args.percent_outliers)

    # Load the NIfTI file
    print("Loading NIfTI file...")
    nii = nib.load(args.nii_path)  # type: ignore
    sample = torch.from_numpy(nii.get_fdata()).unsqueeze(0)  # type: ignore
    sample = MRISample(image=sample)

    # Apply the transforms
    print("Applying transforms...")
    transformed_sample = transforms(sample)

    # Save the transformed NIfTI file
    print("Saving transformed NIfTI file...")
    transformed_nii = nib.Nifti1Image(  # type: ignore
        transformed_sample["image"].numpy(),
        affine=nii.affine,  # type: ignore
    )
    nib.save(transformed_nii, args.out_path)  # type: ignore
