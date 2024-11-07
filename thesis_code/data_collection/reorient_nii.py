import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def reorient_nii_to_ras(nii):
    # Get the affine matrix
    affine = nii.affine

    # Desired RAS+ orientation
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))

    # Get the orientation of the image in matrix form
    current_ornt = nib.orientations.io_orientation(affine)

    # Get the transformation from current orientation to RAS+
    transform_ornt = nib.orientations.ornt_transform(current_ornt, ras_ornt)

    # Apply the orientation transform to the data
    reoriented_data = nib.orientations.apply_orientation(
        nii.get_fdata(), transform_ornt
    ).astype(np.float32)

    # Create a new NIfTI object with the reoriented data and updated affine
    new_affine = nii.affine @ nib.orientations.inv_ornt_aff(transform_ornt, nii.shape)
    reoriented_nii = nib.Nifti1Image(reoriented_data, new_affine)  # type: ignore

    return reoriented_nii


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorient a NIfTI file to RAS+ orientation.\n\n"
        "This script takes a NIfTI file as input, reorients it to RAS+ orientation, "
        "and saves the reoriented file with a .nii.gz extension.\n\n"
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
    # Print previous orientation
    print(f"Original orientation: {nib.orientations.aff2axcodes(nii.affine)}")  # type: ignore
    reoriented_nii = reorient_nii_to_ras(nii)
    # Print new orientation
    print(f"Original orientation: {nib.orientations.aff2axcodes(nii.affine)}")  # type: ignore
    # Replace the extension of the input file with .nii.gz
    output_path = args.input_path.replace(".mgz", ".nii.gz")
    nib.save(reoriented_nii, output_path)  # type: ignore
