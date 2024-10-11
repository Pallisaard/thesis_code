from functools import reduce
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from dataloading.mri_sample import MRISample


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


class VAE3D(nn.Module):
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
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = in_shape[0]
        self.in_size = in_shape[1:]
        self.out_size = in_shape[1:]
        self.encoder_out_channels_per_block = encoder_out_channels_per_block
        self.decoder_out_channels_per_block = decoder_out_channels_per_block
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.stride = stride

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

    def calculate_loss(self, x, recon_x, mu, log_var, beta=1.0):
        """
        Calculate beta-VAE loss = reconstruction loss + β * KL divergence
        If we set beta=1, it'll be the normal VAE loss with reconstruction.
        """
        # Reconstruction loss (MSE for continuous values)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")

        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss with beta weighting
        total_loss = recon_loss + beta * kld_loss

        # You might want to log these separately for monitoring
        return total_loss, recon_loss, kld_loss

    # For KL annealing - check beta-VAE paper
    def get_beta(self, current_epoch, warmup_epochs=10, max_beta=1.0):
        """
        Calculate beta value with optional annealing
        """
        if warmup_epochs > 0:
            # Gradually increase beta from 0 to max_beta
            beta = min(max_beta * (current_epoch / warmup_epochs), max_beta)
        else:
            beta = max_beta
        return beta


class VAE3DLightningModule(L.LightningModule):
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: MRISample, batch_idx):
        x, _ = batch  # Assuming your dataset returns (image, label)
        recon_x, mu, log_var = self(x)

        # Calculate loss
        loss, recon_loss, kld_loss = self.model.calculate_loss(
            x, recon_x, mu, log_var
        )  # Assuming your model has this method

        # Log losses
        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("kld_loss", kld_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
