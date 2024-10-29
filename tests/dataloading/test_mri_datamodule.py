import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from thesis_code.dataloading.mri_datamodule import MRIDataModule
from thesis_code.dataloading.transforms import MRITransform


@pytest.fixture
def data_module():
    return MRIDataModule(
        data_path="./data",
        batch_size=8,
        num_workers=0,
        transform=MagicMock(spec=MRITransform),
        size_limit=None,
    )


@patch("thesis_code.dataloading.mri_datamodule.get_train_dataset")
@patch("thesis_code.dataloading.mri_datamodule.get_val_dataset")
def test_setup_fit(mock_get_val_dataset, mock_get_train_dataset, data_module):
    mock_get_train_dataset.return_value = MagicMock()
    mock_get_val_dataset.return_value = MagicMock()

    data_module.setup(stage="fit")

    mock_get_train_dataset.assert_called_once_with(
        path="./data",
        transform=data_module.transform,
        size_limit=data_module.size_limit,
    )
    mock_get_val_dataset.assert_called_once_with(
        path="./data",
        transforms=data_module.transform,
        size_limit=data_module.size_limit,
    )


@patch("thesis_code.dataloading.mri_datamodule.get_test_dataset")
def test_setup_test(mock_get_test_dataset, data_module):
    mock_get_test_dataset.return_value = MagicMock()

    data_module.setup(stage="test")

    mock_get_test_dataset.assert_called_once_with(
        path="./data",
        transform=data_module.transform,
        size_limit=data_module.size_limit,
    )


def test_train_dataloader(data_module):
    data_module.mri_train = MagicMock()
    dataloader = data_module.train_dataloader(shuffle=False)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == data_module.batch_size
    # assert dataloader.shuffle is True
    assert dataloader.num_workers == data_module.num_workers


def test_val_dataloader(data_module):
    data_module.mri_val = MagicMock()
    dataloader = data_module.val_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == data_module.batch_size
    # assert dataloader.shuffle is False
    assert dataloader.num_workers == data_module.num_workers


def test_test_dataloader(data_module):
    data_module.mri_test = MagicMock()
    dataloader = data_module.test_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == data_module.batch_size
    # assert dataloader.shuffle is False
    assert dataloader.num_workers == data_module.num_workers
