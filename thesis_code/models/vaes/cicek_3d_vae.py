import abc
from functools import reduce
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
)

from thesis_code.dataloading.mri_sample import MRISample


class AbstractVAE3D(abc.ABC):
    @abc.abstractmethod
    def calculate_loss(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    # class ResNetBlock3D(nn.Module):
    #     def __init__(self, in_channels, out_channels, stride=1, groups=32):
    #         """
    #         3D ResNet block with two convolution layers and a residual connection.

    #         Args:
    #             in_channels (int): Number of input channels
    #             out_channels (int): Number of output channels
    #             stride (int): Stride for the first convolution (default: 1)
    #             downsample (nn.Module): Optional downsampling layer for residual connection
    #         """
    #         super(ResNetBlock3D, self).__init__()

    #         # First convolution layer
    #         self.conv1 = nn.Conv3d(
    #             in_channels=in_channels,
    #             out_channels=out_channels,
    #             kernel_size=3,
    #             stride=stride,
    #             padding=1,
    #             bias=False,
    #         )
    #         # self.bn1 = nn.BatchNorm3d(out_channels)
    #         self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
    #         self.relu = nn.ReLU(inplace=True)

    #         # Second convolution layer
    #         self.conv2 = nn.Conv3d(
    #             in_channels=out_channels,
    #             out_channels=out_channels,
    #             kernel_size=3,
    #             stride=1,
    #             padding=1,
    #             bias=False,
    #         )
    #         # self.bn2 = nn.BatchNorm3d(out_channels)
    #         self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    # def forward(self, x):
    #     # First conv block
    #     out = self.conv1(x)
    #     out = self.gn1(out)
    #     out = self.relu(out)

    #     # Second conv block
    #     out = self.conv2(out)
    #     out = self.gn2(out)

    #     # Add residual connection
    #     out += x
    #     out = self.relu(out)

    #     return out


class ConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_unit(x)
        return x


class ConvEncoderBlock(nn.Module):
    def __init__(self, channels: list[int], pool_size: int = 2):
        super().__init__()
        in_out_channel_tuples = list(zip(channels[:-1], channels[1:]))

        self.conv_block = nn.Sequential(
            *[ConvUnit(in_c, out_c) for in_c, out_c in in_out_channel_tuples],
            nn.MaxPool3d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class VAE3DEncoder(nn.Module):
    def __init__(
        self,
        out_channels_per_block: list[int],
        pool_size: int,
        in_shape: Tuple[int, int, int, int],
    ):
        super().__init__()
        self.in_shape = in_shape
        self.in_channels = in_shape[0]

        # Calculate dimensions after each pooling operation
        n_pools = len(out_channels_per_block)
        x_encoded: int = in_shape[1] // (pool_size**n_pools)  # 4 pooling operations
        y_encoded: int = in_shape[2] // (pool_size**n_pools)
        z_encoded: int = in_shape[3] // (pool_size**n_pools)
        c_encoded = out_channels_per_block[-1]
        self.encoded_space = (c_encoded, x_encoded, y_encoded, z_encoded)
        self.flat_encoded_shape = reduce(lambda x, y: x * y, self.encoded_space, 1)

        # First block from in_channels -> out_channels // 2 -> out_channels
        block_1_out_channels = [
            self.in_channels,
            out_channels_per_block[0] // 2,
            out_channels_per_block[0],
        ]
        block_1: nn.Module = ConvEncoderBlock(block_1_out_channels)

        in_out_channel_tuples = list(
            zip(out_channels_per_block[0:-1], out_channels_per_block[1:])
        )
        # Consecutive blocks from in_channels -> in_channels -> out_channels
        consecutive_blocks: list[nn.Module] = [
            ConvEncoderBlock([in_channel, in_channel, out_channel])
            for in_channel, out_channel in in_out_channel_tuples
        ]
        final_conv = ConvUnit(
            out_channels_per_block[-1],
            out_channels_per_block[-1],
        )

        self.encoder = nn.Sequential(block_1, *consecutive_blocks, final_conv)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.flatten(x)


class ConvDecoderBlock(nn.Module):
    def __init__(
        self,
        channels: list[int],
        kernel_size: int = 2,
        stride: int = 2,
    ):
        super().__init__()
        in_out_channel_tuples = list(zip(channels[:-1], channels[1:]))

        self.conv_block = nn.Sequential(
            nn.ConvTranspose3d(
                channels[0], channels[0], kernel_size=kernel_size, stride=stride
            ),
            *[ConvUnit(in_c, out_c) for in_c, out_c in in_out_channel_tuples],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        return x


class VAE3DDecoder(nn.Module):
    def __init__(
        self,
        out_channels_per_block: list[int],
        in_shape: Tuple[int, int, int, int],
        kernel_size: int = 2,
        stride: int = 2,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.in_size = in_shape[1:]
        self.out_channels_per_block = out_channels_per_block
        self.kernel_size = kernel_size
        self.stride = stride

        first_conv = ConvUnit(self.in_channels, self.in_channels)

        in_out_channel_tuples = list(
            zip(out_channels_per_block[0:-1], out_channels_per_block[1:])
        )

        consecutive_blocks: list[nn.Module] = [
            ConvDecoderBlock(
                [in_channel, in_channel, out_channel],
                kernel_size=kernel_size,
                stride=stride,
            )
            for in_channel, out_channel in in_out_channel_tuples
        ]

        self.decoder = nn.Sequential(first_conv, *consecutive_blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        z = z.view(batch_size, *self.in_shape)
        return self.decoder(z)


class VAE3D(nn.Module, AbstractVAE3D):
    def __init__(
        self,
        in_shape: Tuple[int, int, int, int],
        encoder_out_channels_per_block: list[int],
        decoder_out_channels_per_block: list[int],
        latent_dim: int,
        pool_size: int = 2,
        kernel_size: int = 2,
        stride: int = 2,
        beta: float = 1.0,
        beta_annealing: Literal["monotonic", "constant"] = "monotonic",
        max_beta: float = 1.0,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = in_shape[0]
        self.in_size = in_shape[1:]
        self.out_size = in_shape[1:]
        self.encoder_out_channels_per_block = encoder_out_channels_per_block
        self.decoder_out_channels_per_block = decoder_out_channels_per_block
        self.latent_dim = latent_dim
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs

        self.encoder = VAE3DEncoder(encoder_out_channels_per_block, pool_size, in_shape)
        self.decoder = VAE3DDecoder(
            decoder_out_channels_per_block,
            self.encoder.encoded_space,
            kernel_size=kernel_size,
            stride=stride,
        )

        # VAE latent space
        flattened_size = self.encoder.flat_encoded_shape
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_var = nn.Linear(flattened_size, latent_dim)

        # Decoder input
        self.latent_to_decoder = nn.Linear(latent_dim, flattened_size)

    def encode(self, x):
        x = self.encoder(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Get latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.latent_to_decoder(z)
        x = self.decoder(z)

        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        # Remove sigmoid from decoder if not already done
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def calculate_loss(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate beta-VAE loss = reconstruction loss + β * KL divergence
        If we set beta=1, it'll be the normal VAE loss with reconstruction.
        """
        # Reconstruction loss (MSE for continuous values)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence loss
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        beta = torch.tensor(self.get_beta(epoch))

        # Total loss with beta weighting
        total_loss = recon_loss + beta * kld_loss

        # You might want to log these separately for monitoring
        return total_loss, recon_loss, kld_loss, beta

    def get_beta(self, epoch: int):
        if self.beta_annealing == "constant":
            return self.beta
        elif self.beta_annealing == "monotonic":
            return self.monotonic_beta_annealing(epoch)

    # For KL annealing - check beta-VAE paper
    def monotonic_beta_annealing(self, current_epoch):
        """
        Calculate beta value with optional annealing
        """
        if self.warmup_epochs > 0:
            # Gradually increase beta from 0 to max_beta
            beta = min(
                self.max_beta * (current_epoch / self.warmup_epochs), self.max_beta
            )
        else:
            beta = self.max_beta
        return beta


class LitVAE3D(L.LightningModule):
    def __init__(
        self,
        in_shape: Tuple[int, int, int, int],
        encoder_out_channels_per_block: list[int],
        decoder_out_channels_per_block: list[int],
        latent_dim: int,
        pool_size: int = 2,
        kernel_size: int = 2,
        stride: int = 2,
    ):
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters()

        self.model = VAE3D(
            in_shape,
            encoder_out_channels_per_block,
            decoder_out_channels_per_block,
            latent_dim,
            pool_size,
            kernel_size,
            stride,
        )
        self.ssim = StructuralSimilarityIndexMeasure(
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: MRISample, batch_idx: int):
        x = batch["image"]  # Assuming your dataset returns (image, label)
        recon_x, mu, log_var = self(x)

        # Calculate loss
        loss, recon_loss, kld_loss, beta = self.model.calculate_loss(
            x, recon_x, mu, log_var, self.current_epoch + 1
        )  # Assuming your model has this method

        # Log losses
        self.log("train_total_loss", loss, sync_dist=True)
        self.log("train_recon_loss", recon_loss, sync_dist=True)
        self.log("train_kld_loss", kld_loss, sync_dist=True)
        self.log("beta", beta, sync_dist=True)

        return loss

    def validation_step(self, batch: MRISample, batch_idx: int):
        x = batch["image"]
        recon_x, mu, log_var = self(x)

        loss, recon_loss, kld_loss, beta = self.model.calculate_loss(
            x, recon_x, mu, log_var, self.current_epoch + 1
        )

        self.ssim(recon_x, x)

        self.log("val_total_loss", loss, sync_dist=True)
        self.log("val_recon_loss", recon_loss, sync_dist=True)
        self.log("val_kld_loss", kld_loss, sync_dist=True)
        self.log("beta", beta, sync_dist=True)
        self.log("val_ssim", self.ssim, sync_dist=True)

    def test_step(self, batch: MRISample, batch_idx: int):
        x = batch["image"]
        recon_x, mu, log_var = self(x)

        loss, recon_loss, kld_loss, beta = self.model.calculate_loss(
            x, recon_x, mu, log_var, self.current_epoch + 1
        )

        self.ssim(recon_x, x)

        self.log("test_total_loss", loss, sync_dist=True)
        self.log("test_recon_loss", recon_loss, sync_dist=True)
        self.log("test_kld_loss", kld_loss, sync_dist=True)
        self.log("beta", beta, sync_dist=True)
        self.log("test_ssim", self.ssim, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
