from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
import torch
import numpy as np
from numpy.typing import NDArray

from thesis_code.dataloading import MRIDataset


@pytest.fixture
def mock_nifti_data() -> NDArray:
    return np.ones((1, 10, 10, 10), dtype=np.float32)


@pytest.fixture
def mock_nifti_file(tmp_path: Path) -> Path:
    file = tmp_path / "test_image.nii.gz"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()
    return file


@pytest.fixture
def mock_load_nifti():
    with patch("thesis_code.dataloading.mri_dataset.load_nifti") as mock_load:
        nifti = np.ones((10, 10, 10), dtype=np.float32)
        mock_load.return_value = torch.from_numpy(nifti)
        yield mock_load


@pytest.fixture
def mri_dataset(mock_nifti_file, mock_load_nifti, tmp_path):
    return MRIDataset(str(tmp_path))


def test_dataset_initialization(mri_dataset, mock_nifti_file):
    assert len(mri_dataset) == 1
    assert isinstance(mri_dataset.samples, List)


def test_dataset_getitem(mri_dataset, mock_nifti_file, mock_nifti_data):
    sample = mri_dataset[0]

    assert isinstance(sample, torch.Tensor)
    assert sample.shape == torch.Size(mock_nifti_data.shape)
    assert sample.dtype == torch.float32
    assert torch.all(sample == 1.0)


def test_dataset_len(mri_dataset):
    assert len(mri_dataset) == 1


def test_dataset_iteration(mri_dataset):
    for sample in mri_dataset:
        assert isinstance(sample, torch.Tensor)


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
