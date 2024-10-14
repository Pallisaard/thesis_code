import torch
from thesis_code.metrics import ssi_3d, batch_ssi_3d


def test_ssi_3d_single_value_low_res():
    # Create small 3D volumes with identical values
    vol1 = torch.ones(1, 3, 4, 4).float()
    vol1[0, 0, 0, 0] = 0.5
    vol2 = torch.ones(1, 3, 4, 4).float()
    vol2[0, 0, 0, 0] = 0.5

    # Calculate SSIM (should be 1.0 for identical volumes)
    ssim_value = ssi_3d(vol1, vol2, window_size=3)
    print(f"ssim_value: {ssim_value}")
    assert torch.isclose(ssim_value, torch.tensor(1.0), atol=1e-6)


def test_ssi_3d_single_value_high_res():
    # Create larger 3D volumes with identical values
    vol1 = torch.ones(1, 32, 64, 64)
    vol1[0, 0, 0, 0] = 0.5
    vol2 = torch.ones(1, 32, 64, 64)
    vol2[0, 0, 0, 0] = 0.5

    # Calculate SSIM (should be 1.0)
    ssim_value = ssi_3d(vol1, vol2, window_size=7)
    assert torch.isclose(ssim_value, torch.tensor(1.0), atol=1e-6)


def test_ssi_3d_different_values():
    # Create volumes with different values
    vol1 = torch.rand(1, 16, 32, 32)
    vol2 = torch.rand(1, 16, 32, 32)

    # Calculate SSIM (should be less than 1.0)
    ssim_value = ssi_3d(vol1, vol2, window_size=5)
    assert ssim_value < 1.0


def test_ssi_3d_known_value():
    # Create specific volumes with known SSIM (this requires manual calculation or
    # using a reference implementation to determine the expected SSIM)
    vol1 = (
        torch.arange(0, 5**3).float().div(10.0).view((1, 1, 5, 5, 5))
    )  # Example small volume
    vol2 = (
        torch.arange(0, 5**3).float().add(10).div(10.0).view((1, 1, 5, 5, 5))
    )  # Slightly shifted values

    # Calculate SSIM
    ssim_value = ssi_3d(vol1, vol2, window_size=3)
    print("Computed SSI:", ssim_value.item())

    # Replace with the actual expected SSIM value for these volumes
    expected_ssim = 0.9631134867668152  # Example value (calculate this beforehand)
    assert torch.isclose(ssim_value, torch.tensor(expected_ssim), atol=1e-4)


def test_ssi_3d_batched():
    # Create batched 3D volumes with identical values
    batch_size = 8
    vol1 = torch.ones(batch_size, 1, 16, 16, 16).float()
    vol2 = torch.ones(batch_size, 1, 16, 16, 16).float()

    # Introduce slight differences in each volume
    for i in range(batch_size):
        vol1[i, 0, i, i, i] = 0.5
        vol2[i, 0, i, i, i] = 0.5

    # Calculate SSIM for each volume in the batch
    ssim_values = batch_ssi_3d(vol1, vol2, window_size=3, size_average=False)
    print(f"ssim_values: {ssim_values}")

    # Check that each SSIM value is close to 1.0
    assert torch.allclose(ssim_values, torch.ones(batch_size), atol=1e-6)
