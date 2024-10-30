from pathlib import Path

from thesis_code.dataloading.mri_dataset import MRIDataset
from thesis_code.dataloading.transforms import MRITransform


def get_val_dataset(
    path: str,
    transforms: MRITransform | None = None,
    size_limit: int | None = None,
    strip_skulls: bool = False,
) -> MRIDataset:
    test_path = Path(path) / "val"
    return MRIDataset(
        test_path,
        transform=transforms,
        size_limit=size_limit,
        strip_skulls=strip_skulls,
    )


def get_train_dataset(
    path: str,
    transform: MRITransform | None = None,
    size_limit: int | None = None,
    strip_skulls: bool = False,
) -> MRIDataset:
    train_path = Path(path) / "train"
    return MRIDataset(
        train_path,
        transform=transform,
        size_limit=size_limit,
        strip_skulls=strip_skulls,
    )


def get_test_dataset(
    path: str,
    transform: MRITransform | None = None,
    size_limit: int | None = None,
    strip_skulls: bool = True,
) -> MRIDataset:
    train_path = Path(path) / "test"
    return MRIDataset(
        train_path,
        transform=transform,
        size_limit=size_limit,
        strip_skulls=strip_skulls,
    )


def get_mri_dataset(
    path: str,
    transform: MRITransform | None = None,
    size_limit: int | None = None,
    strip_skulls: bool = True,
) -> MRIDataset:
    return MRIDataset(
        path, transform=transform, size_limit=size_limit, strip_skulls=strip_skulls
    )
