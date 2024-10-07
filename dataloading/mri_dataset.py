from pathlib import Path
from typing import TypedDict
from collections.abc import Callable

import torch
from torch.utils.data import Dataset
import nibabel as nib

from mri_sample import MRISample


class MRIDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        transform: Callable[[MRISample], MRISample] | None = None,
    ):
        self.data_path: Path = Path(data_path)
        self.name: str = self.data_path.name
        self.samples: list[MRISample] = self._load_dataset(self.data_path)
        self.transform = transform

    def _load_dataset(self, data_path: Path) -> list[MRISample]:
        scans_dir = data_path / "scans"
        if not scans_dir.exists():
            raise ValueError(f"Scans directory not found in {data_path}")

        samples: list[MRISample] = []
        for i, file in enumerate(scans_dir.glob("**/*.nii.gz")):
            print(f"loading file {i}:{file}")
            img = nib.load(str(file))  # type: ignore
            img_data = img.get_fdata()  # type: ignore
            tensor_data = torch.from_numpy(img_data)
            sample: MRISample = {"image": tensor_data}

            if self.transform is not None:
                sample = self.transform(sample)

            samples.append(sample)

        print(f"{len(samples)} MRI images loaded.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MRISample:
        return self.samples[idx]

    def __repr__(self) -> str:
        return f"MRIDataset({self.name}, {self.data_path})"


def get_val_dataset(path: str) -> MRIDataset:
    test_path = Path(path) / "val"
    return MRIDataset(test_path)


def get_train_dataset(path: str) -> MRIDataset:
    train_path = Path(path) / "train"
    return MRIDataset(train_path)


def get_mri_dataset(path: str) -> MRIDataset:
    return MRIDataset(path)


# Example usage:
if __name__ == "__main__":
    dataset = MRIDataset("../data/ds000140-distinct-brain-systems-extracted")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
