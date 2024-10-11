import lightning as L
from torch.utils.data import DataLoader

from dataloading.mri_dataloader import get_val_dataset, get_train_dataset
from dataloading.transforms import MRITransform


class MRIDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 8,
        num_workers: int = 0,
        transform: MRITransform | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.mri_train = get_train_dataset(
                path=self.data_dir, transform=self.transform
            )
            self.mri_val = get_val_dataset(
                path=self.data_dir, transforms=self.transform
            )
        else:
            raise ValueError(
                "this dataset only supports the fit (train + validate) stages."
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mri_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mri_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
