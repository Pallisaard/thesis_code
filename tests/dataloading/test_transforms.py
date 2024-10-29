import torch
import pytest
from unittest.mock import MagicMock

# Import the necessary classes and types
from thesis_code.dataloading import MRISample
from thesis_code.dataloading import Compose, Resize, ZScoreNormalize
from thesis_code.dataloading import MRIDataset


# Helper function to create a sample image
@pytest.fixture
def sample_image() -> MRISample:
    image = torch.randn(1, 10, 20, 30)
    return {"image": image}


# Mock the MRIDataset class
@pytest.fixture
def dataset() -> MagicMock:
    dataset = MagicMock(spec=MRIDataset)
    dataset.__len__.return_value = 10
    dataset.__getitem__.return_value = {"image": torch.randn(10, 20, 30)}
    return dataset


@pytest.mark.timeout(5)
def test_resize(sample_image: MRISample):
    resize = Resize(size=64)
    resized_sample = resize(sample_image)
    assert resized_sample["image"].shape == (1, 64, 64, 64)


# @pytest.mark.timeout(5)
# def test_zscore_normalize(dataset: MagicMock, sample_image: MRISample):
#     zscore_norm = ZScoreNormalize()
#     zscore_norm.fit(dataset)

#     normalized_sample = zscore_norm(sample_image)
#     denormalized_sample = zscore_norm.denormalize(normalized_sample)

#     # Check that the denormalized sample is close to the original sample
#     assert torch.allclose(denormalized_sample["image"], sample_image["image"])


@pytest.mark.timeout(5)
def test_zscore_normalize_from_params(sample_image: MRISample):
    zscore_norm = ZScoreNormalize.from_parameters(mean=0.0, std=1.0)
    normalized_sample = zscore_norm(sample_image)
    assert abs(normalized_sample["image"].mean().item()) < 0.2  # pytest.approx(0.0)
    assert abs(normalized_sample["image"].std().item()) - 1 < 0.2  # pytest.approx(1.0)


@pytest.mark.timeout(5)
def test_zscore_normalize_save_load():
    zscore_norm = ZScoreNormalize.from_parameters(mean=10.0, std=2.0)
    path = "zscore_params.txt"
    zscore_norm.save(path)
    loaded_zscore_norm = ZScoreNormalize.load_from_disk(path)
    assert loaded_zscore_norm.mean == zscore_norm.mean
    assert loaded_zscore_norm.std == zscore_norm.std


# @pytest.mark.timeout(5)
def test_compose(sample_image: MRISample):
    resize = Resize(size=64)
    zscore_norm = ZScoreNormalize.from_parameters(mean=0.0, std=1.0)
    compose = Compose([resize, zscore_norm])

    transformed_sample = compose(sample_image)
    assert transformed_sample["image"].shape == (1, 64, 64, 64)
    assert abs(transformed_sample["image"].mean().item()) < 0.2  # pytest.approx(0.0)
    assert abs(transformed_sample["image"].std().item()) - 1 < 0.2  # pytest.approx(1.0)
