from pathlib import Path
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from thesis_code.dataloading.utils import load_nifti
from thesis_code.dataloading.transforms import MRITransform


class MRISingleExampleDataset(Dataset):
    """
    A dataset that contains a single MRI example. Used for validation when only
    a single example is needed, such as when using it to generate end of epoch
    examples.
    """

    def __init__(
        self,
        mri: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.mri = mri
        self.transform = transform

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _idx: int) -> torch.Tensor:
        sample = self.mri

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class MRIDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        size_limit: int | None = None,
    ):
        self.data_path: Path = Path(data_path)
        self.name: str = self.data_path.name
        self.transform = transform
        self.size_limit = size_limit

        self.samples: list[Path] = self._load_dataset(self.data_path)

    def _load_dataset(self, data_path: Path) -> list[Path]:
        if not data_path.exists():
            raise ValueError(f"directory doesn't exist: {data_path}")

        samples: list[Path] = list(data_path.rglob("*.nii.gz"))
        if self.size_limit is not None:
            samples = samples[: self.size_limit]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.samples[idx]
        sample = load_nifti(file_path).unsqueeze(0).float()

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self) -> str:
        return f"MRIDataset({self.name}, {self.data_path})"


def get_val_dataset(
    path: str,
    transforms: MRITransform | None = None,
    size_limit: int | None = None,
) -> MRIDataset:
    """
    Returns the dataset at path/val
    """
    test_path = Path(path) / "val"
    return MRIDataset(
        test_path,
        transform=transforms,
        size_limit=size_limit,
    )


def get_train_dataset(
    path: str,
    transform: MRITransform | None = None,
    size_limit: int | None = None,
) -> MRIDataset:
    """
    Returns the dataset at path/train
    """
    train_path = Path(path) / "train"
    return MRIDataset(
        train_path,
        transform=transform,
        size_limit=size_limit,
    )


def get_test_dataset(
    path: str,
    transform: MRITransform | None = None,
    size_limit: int | None = None,
) -> MRIDataset:
    """
    Returns the dataset at path/test
    """
    train_path = Path(path) / "test"
    return MRIDataset(
        train_path,
        transform=transform,
        size_limit=size_limit,
    )


def get_mri_dataset(
    path: str,
    transform: MRITransform | None = None,
    size_limit: int | None = None,
) -> MRIDataset:
    """
    Returns the dataset at path
    """
    return MRIDataset(
        path,
        transform=transform,
        size_limit=size_limit,
    )


def get_single_example_dataset(
    mri: torch.Tensor,
    transform: MRITransform | None = None,
) -> MRISingleExampleDataset:
    return MRISingleExampleDataset(
        mri=mri,
        transform=transform,
    )
