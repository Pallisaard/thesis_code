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
    ):
        super().__init__()
        in_out_channel_tuples = list(zip(channels[:-1], channels[1:]))

        self.conv_block = nn.Sequential(
            nn.ConvTranspose3d(channels[0], channels[0], kernel_size=2, stride=1),
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
    ):
        super().__init__()
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.in_size = in_shape[1:]
        self.out_channels_per_block = out_channels_per_block

        first_conv = ConvUnit(self.in_channels, self.in_channels)

        in_out_channel_tuples = list(
            zip(out_channels_per_block[0:-1], out_channels_per_block[1:])
        )

        consecutive_blocks: list[nn.Module] = [
            ConvDecoderBlock(
                [in_channel, in_channel, out_channel],
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
        beta_annealing: Literal["monotonic", "constant"] = "monotonic",
        constant_beta: float = 1.0,
        max_beta: float = 4.0,
        warmup_epochs: int = 25,
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
        self.beta_annealing = beta_annealing
        self.constant_beta = constant_beta
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs

        self.encoder = VAE3DEncoder(encoder_out_channels_per_block, pool_size, in_shape)
        self.decoder = VAE3DDecoder(
            decoder_out_channels_per_block, self.encoder.encoded_space
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Calculate beta-VAE loss = reconstruction loss + β * KL divergence
        If we set beta=1, it'll be the normal VAE loss with reconstruction.
        """
        # Reconstruction loss (MSE for continuous values)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        log_var = log_var + 1e-8

        # KL divergence loss
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        beta = self.get_beta(epoch)

        # Total loss with beta weighting
        total_loss = recon_loss + beta * kld_loss

        # You might want to log these separately for monitoring
        return total_loss, recon_loss, kld_loss, beta

    def get_beta(self, epoch: int) -> float:
        if self.beta_annealing == "constant":
            return self.constant_beta
        elif self.beta_annealing == "monotonic":
            return self.monotonic_beta_annealing(epoch)
        else:
            raise ValueError("beta_annealing must be one of ['monotonic', 'constant']")

    # For KL annealing - check beta-VAE paper
    def monotonic_beta_annealing(self, current_epoch: int) -> float:
        """
        Calculate beta value with optional annealing
        """
        if self.warmup_epochs > 0:
            # Gradually increase beta from 0 to max_beta
            beta = min(
                self.max_beta * (current_epoch / self.warmup_epochs), self.max_beta
            )
        else:
            beta = 0.0
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
        beta_annealing: Literal["monotonic", "constant"] = "monotonic",
        constant_beta: float = 1.0,
        max_beta: float = 4.0,
        warmup_epochs: int = 25,
    ):
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters()

        self.model = VAE3D(
            in_shape=in_shape,
            encoder_out_channels_per_block=encoder_out_channels_per_block,
            decoder_out_channels_per_block=decoder_out_channels_per_block,
            latent_dim=latent_dim,
            pool_size=pool_size,
            beta_annealing=beta_annealing,
            constant_beta=constant_beta,
            max_beta=max_beta,
            warmup_epochs=warmup_epochs,
        )

        # Use xavier initialization for weights
        self.model.apply(self.init_weights)

        self.ssim = StructuralSimilarityIndexMeasure(
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
        )

    def forward(self, x):
        return self.model(x)

    def sample(self, num_samples: int = 1):
        z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
        samples = self.model.decode(z)
        return samples

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch
        recon_x, mu, log_var = self(x)

        # Clip mean and log_var to prevent extreme values
        mu = torch.clamp(mu, min=-10, max=10)  # Adjust bounds as needed
        log_var = torch.clamp(log_var, min=-10, max=10)  # Adjust bounds as needed
        log_var = log_var + 1e-8  # Add epsilon for numerical stability

        # Calculate loss
        loss, recon_loss, kld_loss, beta = self.model.calculate_loss(
            x, recon_x, mu, log_var, self.current_epoch + 1
        )  # Assuming your model has this method

        # Log losses
        self.log("train_total_loss", loss, sync_dist=True)
        self.log("train_recon_loss", recon_loss, sync_dist=True)
        self.log("train_kld_loss", kld_loss, sync_dist=True)
        self.log("beta", beta)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch
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

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
