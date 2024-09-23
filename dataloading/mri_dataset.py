from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import Dataset
import nibabel as nib


class MRISample(TypedDict):
    image: torch.Tensor
    filename: str


class MRIDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.name = self.data_path.name
        self.samples: list[MRISample] = []

        self._load_dataset()

    def _load_dataset(self):
        scans_dir = self.data_path / "scans"
        if not scans_dir.exists():
            raise ValueError(f"Scans directory not found in {self.data_path}")

        for i, file in enumerate(scans_dir.glob("**/*.nii.gz")):
            print(f"loading file {i}:{file}")
            img = nib.load(str(file))  # type: ignore
            img_data = img.get_fdata()  # type: ignore
            tensor_data = torch.from_numpy(img_data)
            self.samples.append({"image": tensor_data, "filename": str(file)})

        print(f"Loaded {len(self.samples)} MRI images from {self.name} dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MRISample:
        return self.samples[idx]

    def __repr__(self) -> str:
        return f"MRIDataset({self.name}, {self.data_path})"


# Example usage:
if __name__ == "__main__":
    dataset = MRIDataset("../data/ds000140-distinct-brain-systems-extracted")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample filename: {sample['filename']}")
