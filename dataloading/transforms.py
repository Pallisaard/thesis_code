from typing import Sequence

from torch.nn import functional as F
import numpy as np

from dataloading.mri_sample import MRISample
from dataloading.mri_dataset import MRIDataset


class MRITransform:
    def __call__(self, sample: MRISample) -> MRISample:
        raise NotImplementedError


class Compose(MRITransform):
    def __init__(self, transforms: Sequence[MRITransform]):
        self.transforms = transforms

    def __call__(self, sample: MRISample) -> MRISample:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize(MRITransform):
    def __init__(self, size: int):
        self.size = (size, size, size)

    def __call__(self, sample: MRISample) -> MRISample:
        image = sample["image"]
        resized_image = F.interpolate(
            image.unsqueeze(0).unsqueeze(0).float(),  # Add batch and channel dimensions
            size=self.size,
            mode="trilinear",
            align_corners=True,
        )
        sample["image"] = resized_image.squeeze(0).squeeze(
            0
        )  # Remove batch and channel dimensions
        return sample


class ZScoreNormalize(MRITransform):
    def __init__(self):
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, dataset: MRIDataset):
        images = np.stack([sample["image"].numpy() for sample in dataset])
        self.mean = images.mean()
        self.std = images.std()

    def __call__(self, sample: MRISample) -> MRISample:
        if self.mean is None or self.std is None:
            raise ValueError("ZScoreNormalize must be fit to a dataset first.")

        image = sample["image"]
        normalized_image = (image - self.mean) / self.std
        sample["image"] = normalized_image
        return sample

    def denormalize(self, sample: MRISample) -> MRISample:
        if self.mean is None or self.std is None:
            raise ValueError("ZScoreNormalize must be fit to a dataset first.")

        image = sample["image"]
        denormalized_image = (image * self.std) + self.mean
        sample["image"] = denormalized_image
        return sample

    def save(self, path: str):
        if self.mean is None or self.std is None:
            raise ValueError("ZScoreNormalize must be fit to a dataset first.")

        with open(path, "w") as f:
            f.write(f"{self.mean}\n")
            f.write(f"{self.std}\n")

    @classmethod
    def load_from_disk(cls, path: str) -> "ZScoreNormalize":
        try:
            with open(path, "r") as f:
                mean = float(f.readline())
                std = float(f.readline())
        except FileNotFoundError:
            # This error is thrown if the file cannot be found
            raise FileNotFoundError(f"The file at path {path} was not found.")
        except ValueError:
            # This error is thrown if the file contents cannot be converted to float
            raise ValueError(f"The file at path {path} contains invalid data.")

        return cls.from_parameters(mean, std)

    @classmethod
    def from_parameters(cls, mean: float, std: float) -> "ZScoreNormalize":
        norm = cls()
        norm.mean = mean
        norm.std = std
        return norm
