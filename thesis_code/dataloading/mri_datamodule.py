import lightning as L
from torch.utils.data import DataLoader

from thesis_code.dataloading.mri_dataset import (
    get_val_dataset,
    get_train_dataset,
    get_test_dataset,
    get_mri_dataset,
)
from thesis_code.dataloading.transforms import MRITransform


class MRIDataModule(L.LightningDataModule):
    """
    Data Module that uses the standard train/val/test split defined using the data collection script.
    """

    def __init__(
        self,
        data_path: str = "./data",
        batch_size: int = 8,
        num_workers: int = 0,
        transform: MRITransform | None = None,
        size_limit: int | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.size_limit = size_limit

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.mri_train = get_train_dataset(
                path=self.data_path,
                transform=self.transform,
                size_limit=self.size_limit,
            )
            self.mri_val = get_val_dataset(
                path=self.data_path,
                transforms=self.transform,
                size_limit=self.size_limit,
            )
        elif stage == "test":
            self.mri_test = get_test_dataset(
                path=self.data_path,
                transform=self.transform,
                size_limit=self.size_limit,
            )
        else:
            raise ValueError(
                "this dataset only supports the fit (train + validate) stages."
            )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.mri_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mri_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mri_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MRIAllTrainDataModule(L.LightningDataModule):
    """
    Data Module for training on all data. Uses validation data for validation.
    """

    def __init__(
        self,
        data_path: str = "./data",
        batch_size: int = 8,
        num_workers: int = 0,
        transform: MRITransform | None = None,
        size_limit: int | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.size_limit = size_limit

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.mri_train = get_mri_dataset(
                path=self.data_path,
                transform=self.transform,
                size_limit=self.size_limit,
            )
            self.mri_val = get_val_dataset(
                path=self.data_path,
                transforms=self.transform,
                size_limit=self.size_limit,
            )
        else:
            raise ValueError(
                "this dataset only supports the fit (train + validate) stages."
            )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.mri_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mri_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
