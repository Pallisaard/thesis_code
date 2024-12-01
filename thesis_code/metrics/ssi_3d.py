from functools import partial
import torch
import torch.nn.functional as F

from .utils import normalize_to_01, create_3d_window


def ssi_3d(
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Calculates the Structural Similarity Index Measure (SSIM) between two 3D volumes.

    Args:
        vol1: The first 3D volume (torch.Tensor).
        vol2: The second 3D volume (torch.Tensor).
        window_size: The size of the Gaussian kernel.
        size_average: If True, returns the average SSIM across the volume.
                  If False, returns a SSIM map.

    Returns:
        The SSIM value or SSIM map.
    """

    # Normalize volumes to [0, 1]
    vol1, vol2 = normalize_to_01([vol1, vol2])
    # vol2 = normalize_to_01(vol2)

    assert vol1.size() == vol2.size()

    # Create 3D Gaussian window
    window = create_3d_window(window_size).to(vol1.device)

    # Calculate local statistics using 3D convolution
    mu1 = F.conv3d(vol1, window, padding=window_size // 2)
    mu2 = F.conv3d(vol2, window, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(vol1 * vol1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv3d(vol2 * vol2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv3d(vol1 * vol2, window, padding=window_size // 2) - mu1_mu2

    # Check the notes for this, by setting alpha, beta, gamma = 1 it can be reduced to this.
    C1 = 0.01**2
    C2 = 0.03**2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret


def batch_ssi_3d(
    vols1: torch.Tensor,
    vols2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    assert (
        vols1.size() == vols2.size()
    ), f"Input volumes must have the same shape, but got {vols1.size()} and {vols2.size()}"
    assert len(vols1.size()) == 5, "Input volumes must have shape (B, C, D, H, W)"

    _batch_ssi_3d = torch.vmap(
        partial(ssi_3d, window_size=window_size, size_average=size_average)
    )

    if reduction == "mean":
        return _batch_ssi_3d(vols1, vols2).mean()

    return _batch_ssi_3d(vols1, vols2).sum()
