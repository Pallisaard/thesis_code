from typing import Dict, List
from torch.nn import functional as F
import torch
import torch.nn as nn
from opacus.grad_sample import register_grad_sampler
from opacus.utils.tensor_utils import unfold3d
import numpy as np


class SnLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_power_iters=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_power_iters = num_power_iters

        # These are trainable parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize singular vectors for power iteration
        # These are non-trainable buffers for power iteration
        self.u = nn.Parameter(torch.randn(out_features, 1), requires_grad=False)
        self.v = nn.Parameter(torch.randn(in_features, 1), requires_grad=False)

    def _normalize_uv(self):
        self.u.data = F.normalize(self.u.data, dim=0)
        self.v.data = F.normalize(self.v.data, dim=0)

    def _power_iteration(self):
        """Apply power iteration to estimate the spectral norm."""
        with torch.no_grad():
            for _ in range(self.num_power_iters):
                self.v.data = F.normalize(torch.matmul(self.weight.T, self.u), dim=0)
                self.u.data = F.normalize(torch.matmul(self.weight, self.v), dim=0)

    def forward(self, x):
        self._power_iteration()
        sigma_w = torch.matmul(self.u.T, torch.matmul(self.weight, self.v)).squeeze()
        w_sn = self.weight / sigma_w
        out = x @ w_sn.T
        if self.bias is not None:
            out += self.bias
        return out


@register_grad_sampler(SnLinear)
def spectral_norm_linear_grad_sampler(
    module: SnLinear, activations_list: list[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """Computes per-sample gradients for SpectralNormLinear with full correction.

    The gradient follows:
    ∂L/∂W = (1/σ(W)) * [ (backprops ⊗ activations) - (⟨backprops, output⟩ * u v^T) ]

    Args:
        module: SpectralNormLinear instance containing weight W, and singular vectors u, v
        activations: Input tensor x of shape (batch_size, in_features)
        backprops: Gradient tensor dy/dout of shape (batch_size, out_features)
    """
    activations = activations_list[0]

    # Compute spectral norm σ(W) = u^T W v
    sigma_w = torch.matmul(module.u.T, torch.matmul(module.weight, module.v)).squeeze()

    # Per-sample outputs before applying backprops, using the normalized weight
    w_sn = module.weight / sigma_w
    outputs = activations @ w_sn.T
    if module.bias is not None:
        outputs += module.bias

    # Inner product ⟨backprops, outputs⟩ per sample (batch_size,)
    backprops_outputs = (backprops * outputs).sum(dim=1)

    # Correction term: (⟨backprops, outputs⟩ * u v^T)
    uv_outer = module.u @ module.v.T  # (out_features, in_features)
    correction = backprops_outputs.view(-1, 1, 1) * uv_outer.unsqueeze(0)  # (batch_size, out_features, in_features)

    # Standard per-sample gradient (outer product of backprops and activations)
    grad_weight = backprops.unsqueeze(2) * activations.unsqueeze(1)

    # Apply spectral norm scaling and correction
    grad_weight = (grad_weight - correction) / sigma_w

    per_sample_grads = {}
    if module.weight.requires_grad:
        per_sample_grads[module.weight] = grad_weight

    if module.bias is not None and module.bias.requires_grad:
        per_sample_grads[module.bias] = backprops

    return per_sample_grads


class SnConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_power_iters=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.groups = groups
        self.num_power_iters = num_power_iters

        # Initialize weight and optional bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Calculate weight matrix shape for power iteration
        weight_matrix_shape = (
            out_channels,
            (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
        )

        # Initialize singular vectors
        self.u = nn.Parameter(torch.randn(weight_matrix_shape[0], 1), requires_grad=False)
        self.v = nn.Parameter(torch.randn(weight_matrix_shape[1], 1), requires_grad=False)

    def _reshape_weight_matrix(self):
        """Reshape 5D weight tensor to 2D matrix for power iteration."""
        weight_matrix = self.weight.reshape(self.weight.size(0), -1)
        return weight_matrix

    def _normalize_uv(self):
        self.u.data = F.normalize(self.u.data, dim=0)
        self.v.data = F.normalize(self.v.data, dim=0)

    def _power_iteration(self):
        """Apply power iteration to estimate the spectral norm."""
        with torch.no_grad():
            weight_matrix = self._reshape_weight_matrix()
            for _ in range(self.num_power_iters):
                # v = W^T u / ||W^T u||
                self.v.data = F.normalize(torch.matmul(weight_matrix.T, self.u), dim=0)
                # u = W v / ||W v||
                self.u.data = F.normalize(torch.matmul(weight_matrix, self.v), dim=0)

    def forward(self, x):
        self._power_iteration()
        weight_matrix = self._reshape_weight_matrix()

        # Compute spectral norm
        sigma_w = torch.matmul(self.u.T, torch.matmul(weight_matrix, self.v)).squeeze()

        # Normalize weight tensor
        w_sn = self.weight / sigma_w

        # Apply convolution with normalized weights
        out = F.conv3d(x, w_sn, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


@register_grad_sampler(SnConv3d)
@register_grad_sampler(SnConv3d)
def compute_conv_grad_sample(
    layer: SnConv3d,
    activations_list: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per-sample gradients for spectral norm 3D convolutional layers
    with full correction term.
    """
    activations = activations_list[0]
    n = activations.shape[0]
    if n == 0:
        ret = {}
        ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.zeros_like(layer.bias).unsqueeze(0)
        return ret

    # Compute spectral norm σ(W)
    weight_matrix = layer.weight.reshape(layer.weight.size(0), -1)
    sigma_w = torch.matmul(layer.u.T, torch.matmul(weight_matrix, layer.v)).squeeze()

    # Unfold activations to match the convolution operation
    activations_unfolded = unfold3d(
        activations,
        kernel_size=(int(layer.kernel_size[0]), int(layer.kernel_size[1]), int(layer.kernel_size[2])),
        padding=(int(layer.padding[0]), int(layer.padding[1]), int(layer.padding[2])),
        stride=(int(layer.stride[0]), int(layer.stride[1]), int(layer.stride[2])),
        dilation=(int(layer.dilation[0]), int(layer.dilation[1]), int(layer.dilation[2])),
    )  # (n, p, L)
    backprops = backprops.reshape(n, -1, activations_unfolded.shape[-1])  # (n, o, L)

    # Standard per-sample gradient (before correction)
    grad_sample = torch.einsum("noq,npq->nop", backprops, activations_unfolded)

    # Reshape to match weight shape
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        int(np.prod(layer.kernel_size).item()),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    grad_sample = grad_sample.view(n, *layer.weight.shape)

    # --- Correction term ---
    uv_outer = layer.u @ layer.v.T  # (out_channels, in_channels * kernel_size_product)
    uv_outer = uv_outer.view(*layer.weight.shape)  # Reshape to weight shape
    # Inner product ⟨backprops, outputs⟩ per sample
    outputs = F.conv3d(
        activations, layer.weight / sigma_w, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups
    )
    backprops_outputs = (backprops * outputs.reshape_as(backprops)).sum(dim=[1, 2])  # (n,)

    correction = backprops_outputs.view(n, 1, 1, 1, 1, 1) * uv_outer.unsqueeze(0)

    # Apply correction and scale by spectral norm
    grad_sample = (grad_sample - correction) / sigma_w

    ret = {}
    if layer.weight.requires_grad:
        ret[layer.weight] = grad_sample

    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = backprops.sum(dim=2)

    return ret
