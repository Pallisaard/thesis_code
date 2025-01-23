import nibabel as nib


def get_orientation(affine):
    """
    Determine the orientation of the NIfTI file based on its affine matrix.
    Returns a string representing the orientation (e.g., 'RAS', 'LIA', etc.).
    """
    # Get the orientation codes from the affine matrix
    orientation = nib.orientations.aff2axcodes(affine)
    return "".join(orientation)


def main(nii_path):
    # Load the NIfTI file
    img = nib.load(nii_path)  # type: ignore

    # Get the affine matrix
    affine = img.affine  # type: ignore

    # Determine the orientation
    orientation = get_orientation(affine)

    print(f"The orientation of the NIfTI file is: {orientation}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python nii_orientation.py <path_to_nii.gz>")
    else:
        nii_path = sys.argv[1]
        main(nii_path)
