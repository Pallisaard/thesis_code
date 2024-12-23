from typing import Sequence
from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F
import numpy as np


class MRITransform(ABC):
    def __init__(self):
        self.indent = 0

    @abstractmethod
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Sample should be of shape (C, D, H, W)
        """
        ...

    def __repr__(self) -> str:
        """
        Defines a standard way to represent the transform as a string.

        Usually in the form of MRITransformName(arg1=val1, arg2=val2, ...)
        """
        return (
            self.__class__.__name__
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if k != "indent"])
            + ")"
        )

    def transform_np(self, sample: np.ndarray) -> np.ndarray:
        mri_sample = torch.from_numpy(sample).unsqueeze(0)
        transformed_sample = self(mri_sample)
        return transformed_sample.squeeze(0).numpy()


class Compose(MRITransform):
    def __init__(self, transforms: Sequence[MRITransform]):
        super().__init__()
        self.indent += 4
        self.transforms = transforms

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        """
        Overwrites the default repr to print a composition of transforms.
        """
        indentation = " " * self.indent
        return (
            self.__class__.__name__
            + "(\n"
            + indentation
            + (",\n" + indentation).join([str(t) for t in self.transforms])
            + "\n)"
        )


class RemoveZeroSlices(MRITransform):
    def __init__(self):
        super().__init__()

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Input has shape (C, D, H, W)
        # Check if the sample is 3D
        if sample.dim() != 4:
            raise ValueError("Input sample must be a 3D tensor")

        # Remove slices along the depth axis (axis 1)
        mask_depth = (sample != 0).any(dim=(0, 2, 3))
        sample = sample[:, mask_depth, :, :]

        # Remove slices along the height axis (axis 2)
        mask_height = (sample != 0).any(dim=(0, 1, 3))
        sample = sample[:, :, mask_height, :]

        # Remove slices along the width axis (axis 3)
        mask_width = (sample != 0).any(dim=(0, 1, 2))
        sample = sample[:, :, :, mask_width]

        return sample


class Identity(MRITransform):
    def __init__(self):
        super().__init__()

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return sample


class Resize(MRITransform):
    def __init__(self, size: int):
        super().__init__()
        self.size = (size, size, size)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.shape[1:] == self.size:
            return sample
        resized_sample = F.interpolate(
            sample.unsqueeze(0),  # Add batch dimensions
            size=self.size,
            mode="trilinear",
            align_corners=True,
        ).squeeze(0)  # Remove batch dimension
        return resized_sample


class RemovePercentOutliers(MRITransform):
    def __init__(self, percent: float):
        super().__init__()
        if percent < 0 or percent > 1:
            raise ValueError("percent must be between 0 and 1")

        self.percent = percent

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        sample_copy = sample.clone()  # Create a copy of the sample tensor
        bound = torch.quantile(sample_copy, self.percent, keepdim=True)
        sample_copy[sample_copy > bound] = (
            bound  # Modify the copy instead of the original
        )
        return sample_copy


class RangeNormalize(MRITransform):
    def __init__(self, target_min: float = -1, target_max: float = 1):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Dynamically calculate source range
        source_min = sample.min()
        source_max = sample.max()

        # Avoid division by zero
        if source_min == source_max:
            raise ValueError("Input sample has no intensity variation")

        # Direct scaling to target range
        normalized_sample = (self.target_max - self.target_min) * (
            sample - source_min
        ) / (source_max - source_min) + self.target_min
        return normalized_sample


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
