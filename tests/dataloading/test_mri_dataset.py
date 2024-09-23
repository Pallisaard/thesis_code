from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest
import torch
import numpy as np
from numpy.typing import NDArray

from dataloading.mri_dataset import MRIDataset


@pytest.fixture
def mock_nifti_data() -> NDArray:
    return np.ones((10, 10, 10), dtype=np.float32)


@pytest.fixture
def mock_nifti_file(tmp_path: Path) -> Path:
    file = tmp_path / "scans" / "test_image.nii.gz"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    return file


@pytest.fixture
def mock_nibabel(mock_nifti_data):
    with patch("dataloading.mri_dataset.nib") as mock_nib:
        mock_img = Mock()
        mock_img.get_fdata.return_value = mock_nifti_data
        mock_nib.load.return_value = mock_img
        yield mock_nib


@pytest.fixture
def mri_dataset(tmp_path, mock_nifti_file, mock_nibabel):
    return MRIDataset(str(tmp_path))


def test_dataset_initialization(mri_dataset, mock_nifti_file):
    assert len(mri_dataset) == 1
    assert isinstance(mri_dataset.samples, List)


def test_dataset_getitem(mri_dataset, mock_nifti_file, mock_nifti_data):
    sample = mri_dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == torch.Size(mock_nifti_data.shape)
    assert sample["image"].dtype == torch.float32
    assert torch.all(sample["image"] == 1.0)

    assert isinstance(sample["filename"], str)
    assert sample["filename"] == str(mock_nifti_file)


def test_dataset_len(mri_dataset):
    assert len(mri_dataset) == 1


def test_dataset_iteration(mri_dataset):
    for sample in mri_dataset:
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["filename"], str)


def test_invalid_data_path():
    with pytest.raises(ValueError):
        MRIDataset("/nonexistent/path")


def test_empty_dataset(tmp_path):
    empty_dir = tmp_path / "empty" / "scans"
    empty_dir.mkdir(parents=True, exist_ok=True)
    dataset = MRIDataset(str(tmp_path / "empty"))
    assert len(dataset) == 0


def test_dataset_repr(mri_dataset):
    assert (
        str(mri_dataset) == f"MRIDataset({mri_dataset.name}, {mri_dataset.data_path})"
    )


if __name__ == "__main__":
    pytest.main([__file__])
