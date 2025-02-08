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


def register_dp_layers():
    """Explicitly register DP-compatible layers with Opacus"""

    @register_grad_sampler(SpectralNormLinear)
    def grad_sampler_spectral_norm_linear(layer, activations, backprops):
        ret = {layer.weight: torch.einsum("ni,nj->nij", backprops, activations)}
        if layer.bias is not None:
            ret[layer.bias] = torch.sum(backprops, dim=0)
        return ret

    @register_grad_sampler(SpectralNormConv3d)
    def grad_sampler_spectral_norm_conv3d(layer, activations, backprops):
        # For Conv3d, activations shape: (n, c_in, d, h, w)
        # backprops shape: (n, c_out, d_out, h_out, w_out)
        n = activations.shape[0]

        # Unfold activations
        activations_unf = torch.nn.functional.unfold(
            activations.reshape(n * activations.shape[1], 1, *activations.shape[2:]),
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
        )
        activations_unf = activations_unf.reshape(
            n, layer.in_channels, activations_unf.shape[-2], activations_unf.shape[-1]
        )

        # Reshape backprops
        backprops_reshaped = backprops.reshape(n, layer.out_channels, -1)

        # Calculate weight gradients
        ret = {
            layer.weight: torch.einsum(
                "nkm,nilm->ikl", backprops_reshaped, activations_unf
            ).reshape(layer.weight.shape)
        }

        # Add bias gradients if needed
        if layer.bias is not None:
            ret[layer.bias] = torch.sum(backprops_reshaped, dim=(0, 2))

        return ret


# Call the registration function immediately
register_dp_layers()
