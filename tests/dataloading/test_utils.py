from unittest.mock import patch, MagicMock
import torch
from thesis_code.dataloading.utils import load_nifti


@patch("nibabel.load")
def test_load_nifti(mock_nibabel_load):
    # Create a mock NIfTI image with fake data
    fake_data = torch.rand(10, 10, 10).numpy()
    mock_volume = MagicMock()
    mock_volume.get_fdata.return_value = fake_data
    mock_nibabel_load.return_value = mock_volume

    # Call the function under test
    file_path = "fake_file.nii.gz"
    result = load_nifti(file_path)

    # Check that nib.load was called with the correct file path
    mock_nibabel_load.assert_called_once_with(file_path)

    # Check that the result is a torch.Tensor with the correct data
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, torch.from_numpy(fake_data))
