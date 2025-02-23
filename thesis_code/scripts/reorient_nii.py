import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image
from nilearn.datasets import load_mni152_template


def reorient_nii_to_ras(nii, orientation: str = "RAS"):
    # Get the affine matrix
    affine = nii.affine

    orientation_tuple = tuple(letter for letter in orientation)

    # Desired RAS+ orientation
    new_ornt = nib.orientations.axcodes2ornt(orientation_tuple)

    # Get the orientation of the image in matrix form
    current_ornt = nib.orientations.io_orientation(affine)

    # Get the transformation from current orientation to RAS+
    transform_ornt = nib.orientations.ornt_transform(current_ornt, new_ornt)

    # Apply the orientation transform to the data
    reoriented_data = nib.orientations.apply_orientation(
        nii.get_fdata(), transform_ornt
    ).astype(np.float32)

    # Create a new NIfTI object with the reoriented data and updated affine
    new_affine = nii.affine @ nib.orientations.inv_ornt_aff(transform_ornt, nii.shape)
    reoriented_nii = nib.Nifti1Image(reoriented_data, new_affine)  # type: ignore

    return reoriented_nii


def resample_to_talairach(nii):
    """
    Resample the input NIfTI image to Talairach space using the MNI152 template.
    """
    # Load the MNI152 template (a modern alternative to Talairach space)
    template = load_mni152_template()

    # Resample the input image to the template space
    resampled_nii = image.resample_to_img(
        nii, template, force_resample=True, copy_header=True
    )

    return resampled_nii


def zero_values_below_threshold(nii, threshold=1e-3):
    data = nii.get_fdata()
    data[data < threshold] = 0.0
    nii = nib.Nifti1Image(data, nii.affine)  # type: ignore
    return nii


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorient a NIfTI file to RAS+ orientation and resample to Talairach space.\n\n"
        "This script takes a NIfTI file as input, reorients it to RAS+ orientation, "
        "resamples it to Talairach space (using the MNI152 template), and saves the "
        "reoriented and resampled file with a .nii.gz extension.\n\n"
        "Example usage:\n"
        "  python reorient_nii.py input_file.nii.gz\n"
        "This will produce a file named input_file_reoriented.nii.gz in the same directory."
    )
    parser.add_argument("input_path", type=str, help="Path to the input NIfTI file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # For a single .nii.gz file
    nii_path = Path(args.input_path)
    nii = nib.load(nii_path)  # type: ignore

    # Print original orientation
    print(f"Original orientation: {nib.orientations.aff2axcodes(nii.affine)}")  # type: ignore

    # Reorient to RAS+
    reoriented_nii = reorient_nii_to_ras(nii)

    # Resample to Talairach space (MNI152 template)
    resampled_nii = resample_to_talairach(reoriented_nii)

    # Print new orientation
    print(
        f"New orientation after resampling: {nib.orientations.aff2axcodes(resampled_nii.affine)}"
    )  # type: ignore

    # Replace the extension of the input file with _reoriented_resampled.nii.gz
    output_path = args.input_path.replace(".nii.gz", "_reoriented_resampled.nii.gz")
    nib.save(resampled_nii, output_path)  # type: ignore

    print(f"Reoriented and resampled image saved to: {output_path}")
