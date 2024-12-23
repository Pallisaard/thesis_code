import torch
import pytest
from unittest.mock import MagicMock

# Import the necessary classes and types
from thesis_code.dataloading import MRIDataset
from thesis_code.dataloading.transforms import (
    Resize,
    Compose,
    RangeNormalize,
    RemovePercentOutliers,
    RemoveZeroSlices,
)


# Helper function to create a sample image
@pytest.fixture
def sample_image() -> torch.Tensor:
    image = torch.randn(1, 10, 20, 30)
    return image


@pytest.fixture
def zero_image() -> torch.Tensor:
    image = torch.zeros(1, 10, 10, 10)
    return image


# Mock the MRIDataset class
@pytest.fixture
def dataset() -> MagicMock:
    dataset = MagicMock(spec=MRIDataset)
    dataset.__len__.return_value = 10
    dataset.__getitem__.return_value = {"image": torch.randn(10, 20, 30)}
    return dataset


@pytest.mark.timeout(5)
def test_resize(sample_image: torch.Tensor):
    resize = Resize(size=64)
    resized_sample = resize(sample_image)
    assert resized_sample.shape == (1, 64, 64, 64)


def test_range_normalize(sample_image: torch.Tensor):
    range_norm = RangeNormalize(target_min=0.0, target_max=2.0)
    normalized_sample = range_norm(sample_image)
    assert abs(normalized_sample.min().item()) == 0.0
    assert abs(normalized_sample.max().item()) == 2.0


def test_remove_percent_outliers(sample_image: torch.Tensor):
    remove_outliers = RemovePercentOutliers(percent=0.95)
    cleaned_sample = remove_outliers(sample_image)

    assert cleaned_sample.shape == (1, 10, 20, 30)
    assert sample_image.shape == (1, 10, 20, 30)
    # Find the positions of the highest 5% of elements in sample_image
    threshold = torch.quantile(sample_image, 0.95, keepdim=True)
    high_value_positions = sample_image > threshold

    # Assert that none of the values in high_value_positions are equal in cleaned_sample
    print(sample_image)
    print(sample_image[high_value_positions])
    print(cleaned_sample[high_value_positions])
    assert torch.all(
        sample_image[high_value_positions] != cleaned_sample[high_value_positions]
    )


# @pytest.mark.timeout(5)
def test_compose(sample_image: torch.Tensor):
    resize = Resize(size=64)
    range_norm = RangeNormalize(target_min=-1.0, target_max=1.0)
    compose = Compose([resize, range_norm])

    transformed_sample = compose(sample_image)
    assert transformed_sample.shape == (1, 64, 64, 64)
    assert transformed_sample.min().item() == -1.0
    assert transformed_sample.max().item() == 1.0


def test_remove_zero_slices(zero_image: torch.Tensor):
    zero_image[:, 3:5, 2:6, 1:7] = 1
    remove_zero_slices = RemoveZeroSlices()
    cleaned_sample = remove_zero_slices(zero_image)

    assert cleaned_sample.shape == (1, 2, 4, 6)
