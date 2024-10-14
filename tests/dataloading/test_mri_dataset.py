from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest
import torch
import numpy as np
from numpy.typing import NDArray

from dataloading import MRIDataset, MRISample


@pytest.fixture
def mock_nifti_data() -> NDArray:
    return np.ones((10, 10, 10), dtype=np.float32)


@pytest.fixture
def mock_nifti_file(tmp_path: Path) -> Path:
    file = tmp_path / "test_image.nii.gz"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    return file


@pytest.fixture
def mock_load_nifti(mock_nifti_data):
    with patch("dataloading.mri_dataset.load_nifti") as mock_load:
        mock_load.return_value = torch.from_numpy(mock_nifti_data)
        yield mock_load


@pytest.fixture
def mri_dataset(mock_nifti_file, mock_load_nifti, tmp_path):
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


def test_dataset_len(mri_dataset):
    assert len(mri_dataset) == 1


def test_dataset_iteration(mri_dataset):
    for sample in mri_dataset:
        assert isinstance(sample["image"], torch.Tensor)


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
