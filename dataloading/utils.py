from pathlib import Path
import torch
import nibabel as nib


def load_nifti(file_path: str | Path) -> torch.Tensor:
    img = nib.load(str(file_path))  # type: ignore
    mri_data = img.get_fdata()  # type: ignore
    mri_torch_data = torch.from_numpy(mri_data)
    return mri_torch_data
