from pathlib import Path

from thesis_code.dataloading.mri_dataset import MRIDataset
from thesis_code.dataloading.transforms import MRITransform


def get_val_dataset(path: str, transforms: MRITransform | None = None) -> MRIDataset:
    test_path = Path(path) / "val"
    return MRIDataset(test_path, transform=transforms)


def get_train_dataset(path: str, transform: MRITransform | None = None) -> MRIDataset:
    train_path = Path(path) / "train"
    return MRIDataset(train_path, transform=transform)


def get_mri_dataset(path: str, transform: MRITransform | None = None) -> MRIDataset:
    return MRIDataset(path, transform=transform)
