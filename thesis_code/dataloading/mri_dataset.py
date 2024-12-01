from pathlib import Path
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from thesis_code.dataloading.utils import load_nifti


class MRISingleExampleDataset(Dataset):
    def __init__(
        self,
        mri: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.mri = mri
        self.transform = transform

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
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
        strip_skulls: bool = False,
    ):
        self.data_path: Path = Path(data_path)
        self.name: str = self.data_path.name
        self.transform = transform
        self.size_limit = size_limit
        self.strip_skulls = strip_skulls

        self.samples: list[Path] = self._load_dataset(self.data_path)

    def get_brain_mask(self, brain_mask_path: Path) -> torch.Tensor:
        return load_nifti(self.data_path / brain_mask_path)

    def apply_brain_mask(
        self, mri: torch.Tensor, brain_mask: torch.Tensor
    ) -> torch.Tensor:
        return mri * brain_mask

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
