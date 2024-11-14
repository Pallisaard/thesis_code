from typing import Literal, Sequence
from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F
import numpy as np

from thesis_code.dataloading.mri_sample import MRISample
from thesis_code.dataloading.mri_dataset import MRIDataset
from tqdm import tqdm


class MRITransform(ABC):
    def __init__(self):
        self.indent = 0

    @abstractmethod
    def __call__(self, sample: MRISample) -> MRISample: ...

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if k != "indent"])
            + ")"
        )


class Compose(MRITransform):
    def __init__(self, transforms: Sequence[MRITransform]):
        super().__init__()
        self.indent += 4
        self.transforms = transforms

    def __call__(self, sample: MRISample) -> MRISample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        indentation = " " * self.indent
        return (
            self.__class__.__name__
            + "(\n"
            + indentation
            + (",\n" + indentation).join([str(t) for t in self.transforms])
            + "\n)"
        )


class Resize(MRITransform):
    def __init__(self, size: int):
        super().__init__()
        self.size = (size, size, size)

    def __call__(self, sample: MRISample) -> MRISample:
        image = sample["image"]
        if image.shape[1:] == self.size:
            return sample
        resized_image = F.interpolate(
            image.unsqueeze(0),  # Add batch and channel dimensions
            size=self.size,
            mode="trilinear",
            align_corners=True,
        ).squeeze(0)  # Remove batch dimension
        sample["image"] = resized_image
        return sample


class ZScoreNormalize(MRITransform):
    def __init__(self):
        super().__init__()
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, dataset: MRIDataset, batch_size: int = 32) -> "ZScoreNormalize":
        mean_sum: float = 0.0
        sq_sum: float = 0.0
        num_samples: float = 0.0

        for i in tqdm(
            range(0, len(dataset), batch_size),
            total=len(dataset) // batch_size,
            desc="Fitting ZScoreNormalize",
        ):
            batch_idx = range(i, min(i + batch_size, len(dataset)))
            batch = [dataset[idx] for idx in batch_idx]
            batch_mris = torch.stack([sample["image"] for sample in batch])

            mean_sum += (batch_mris).sum().item()
            sq_sum += (batch_mris**2).sum().item()
            num_samples += len(batch_mris)

        mean: float = mean_sum / num_samples
        std: float = np.sqrt(sq_sum / num_samples - mean**2)

        self.mean = mean
        self.std = std
        return self

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


class RemovePercentOutliers(MRITransform):
    def __init__(self, percent: float):
        super().__init__()
        self.percent = percent

    def __call__(self, sample: MRISample) -> MRISample:
        image = sample["image"]
        abs_image = np.abs(image)
        bound = np.percentile(abs_image, self.percent)
        image[abs_image < bound] = bound  # type: ignore
        sample["image"] = image
        return sample


class RangeNormalize(MRITransform):
    def __init__(self, target_min: float = -1, target_max: float = 1):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max

    def __call__(self, sample: MRISample) -> MRISample:
        image = sample["image"]
        # Dynamically calculate source range
        source_min = image.min()
        source_max = image.max()

        # Avoid division by zero
        if source_min == source_max:
            raise ValueError("Input image has no intensity variation")

        # Direct scaling to target range
        normalized_image = (self.target_max - self.target_min) * (
            image - source_min
        ) / (source_max - source_min) + self.target_min
        sample["image"] = normalized_image
        return sample


def normalize_to_0_1(array: np.ndarray) -> np.ndarray:
    min_val = array.min()
    max_val = array.max()

    if min_val == max_val:
        raise ValueError("Input array has no intensity variation")

    array = (array - min_val) / (max_val - min_val)
    return array


def normalize_to(array: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    min_val = array.min()
    max_val = array.max()

    if min_val == max_val:
        raise ValueError("Input array has no intensity variation")

    array = (target_max - target_min) * (array - min_val) / (
        max_val - min_val
    ) + target_min
    return array
