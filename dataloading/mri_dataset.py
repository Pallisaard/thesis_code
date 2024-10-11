from pathlib import Path
from collections.abc import Callable

import torch
from torch.utils.data import Dataset
import nibabel as nib

from dataloading.mri_sample import MRISample


class MRIDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        transform: Callable[[MRISample], MRISample] | None = None,
    ):
        self.data_path: Path = Path(data_path)
        self.name: str = self.data_path.name
        self.transform = transform
        self.samples: list[Path] = self._load_dataset(self.data_path)

    def _load_dataset(self, data_path: Path) -> list[Path]:
        scans_dir = data_path / "scans"
        if not scans_dir.exists():
            raise ValueError(f"Scans directory not found in {data_path}")

        samples: list[Path] = list(scans_dir.glob("**/*.nii.gz"))
        print(f"{len(samples)} MRI images loaded.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MRISample:
        file = self.samples[idx]
        img = nib.load(str(file))  # type: ignore
        img_data = img.get_fdata()  # type: ignore
        sample: MRISample = {"image": torch.from_numpy(img_data)}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self) -> str:
        return f"MRIDataset({self.name}, {self.data_path})"
