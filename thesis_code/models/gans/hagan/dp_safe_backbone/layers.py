from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from opacus.grad_sample import register_grad_sampler


# Spectral Normalized Linear Layer (DP Compatible)
class SpectralNormLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        spectral_norm(self)  # Apply spectral normalization to weights


# Spectral Normalized Conv3D Layer (DP Compatible)
class SpectralNormConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: int | Tuple[int, int, int] = 1,
        padding: int | Tuple[int, int, int] = 0,
        bias=True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        spectral_norm(self)  # Apply spectral normalization to weights


# Registering per-sample gradient computation for Opacus
@register_grad_sampler(SpectralNormLinear)
def grad_sampler_spectral_norm_linear(layer, activations, backprops):
    return {layer.weight: torch.einsum("ni,nj->nij", backprops, activations)}


@register_grad_sampler(SpectralNormConv3d)
def grad_sampler_spectral_norm_conv3d(layer, activations, backprops):
    return {layer.weight: torch.einsum("nchwk,nchw->nchwk", backprops, activations)}
