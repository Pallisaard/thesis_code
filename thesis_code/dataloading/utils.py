from pathlib import Path
import torch
import numpy as np
import nibabel as nib

from thesis_code.dataloading.transforms import normalize_to


def numpy_to_nifti(array: np.ndarray) -> nib.Nifti1Image:  # type: ignore
    """
    Convert a 3D numpy array to a Nifti1Image with RAS orientation.

    Parameters:
    array (np.ndarray): 3D numpy array to convert.

    Returns:
    nib.Nifti1Image: Nifti image that can be saved as an .nii.gz file.
    """
    # Ensure the array is 3D
    if array.ndim != 3:
        raise ValueError("Input array must be 3D")

    # Create an identity affine matrix
    affine = np.eye(4)

    # Create the Nifti1Image
    nifti_img = nib.Nifti1Image(array, affine)  # type: ignore

    # Set the orientation to RAS
    nifti_img = nib.as_closest_canonical(nifti_img)  # type: ignore

    return nifti_img


def load_nifti(file_path: str | Path) -> torch.Tensor:
    img = nib.load(str(file_path))  # type: ignore
    mri_data = img.get_fdata()  # type: ignore
    return torch.from_numpy(mri_data)


def save_mri(images: torch.Tensor, file_path: Path) -> None:
    true_array = normalize_to(images[0, 0].cpu().numpy(), -1, 1)
    true_nii = numpy_to_nifti(true_array)
    nib.save(true_nii, file_path)  # type: ignore
