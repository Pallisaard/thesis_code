from pathlib import Path
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from thesis_code.dataloading.mri_sample import MRISample
from thesis_code.dataloading.utils import load_nifti


class MRIDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        transform: Callable[[MRISample], MRISample] | None = None,
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
        return load_nifti(self.data_path / "masks" / brain_mask_path)

    def apply_brain_mask(
        self, mri: torch.Tensor, brain_mask: torch.Tensor
    ) -> torch.Tensor:
        return mri * brain_mask

    def _load_dataset(self, data_path: Path) -> list[Path]:
        scans_dir = data_path
        if not scans_dir.exists():
            raise ValueError(f"Scans directory not found in {data_path}")

        samples: list[Path] = list(scans_dir.rglob("*.nii.gz"))
        if self.size_limit is not None:
            samples = samples[: self.size_limit]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MRISample:
        file_path = self.samples[idx]
        mri = load_nifti(file_path).unsqueeze(0).float()
        sample: MRISample = {"image": mri}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self) -> str:
        return f"MRIDataset({self.name}, {self.data_path})"
