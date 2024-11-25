from typing import Literal, Sequence
import torch
import torch.nn as nn
from monai.networks.nets.resnet import resnet10, resnet50


def get_mri_vectorizer(model_type: Literal[10, 50]) -> nn.Module:
    """Get MRI vectorizer model."""
    if model_type == 10:
        return resnet10(n_input_channels=1, feed_forward=False)
    elif model_type == 50:
        return resnet50(n_input_channels=1, feed_forward=False)
    else:
        raise ValueError("Invalid model type.")


def normalize_to_01(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Normalizes a sequence of tensors to the range [0, 1]."""
    max_val = torch.max(torch.stack([t.max() for t in tensors]))
    min_val = torch.min(torch.stack([t.min() for t in tensors]))
    # assert max_val != min_val, "Cannot normalize tensor with min == max"
    return [(t - min_val) / (max_val - min_val) for t in tensors]


def gaussian_pdf(x: int, window_size: int, sigma: float) -> float:
    return -((x - window_size // 2) ** 2) / float(2 * sigma**2)


def gaussian_3d(window_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    gauss = torch.Tensor(
        [gaussian_pdf(x, window_size, sigma) for x in range(window_size)]
    )
    gauss = torch.exp(gauss)
    return gauss / gauss.sum()


def create_3d_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """Create 3D Gaussian window."""
    _1D_window = gaussian_3d(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return _3D_window
