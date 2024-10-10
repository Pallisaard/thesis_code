from typing import Tuple

import torch
import torch.nn.functional as F

import lightning.pytorch as L


from models.cicek_3d_vae import VAE3D
from dataloading.mri_dataset import MRISample


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
