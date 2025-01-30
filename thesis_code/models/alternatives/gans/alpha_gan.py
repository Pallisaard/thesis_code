import time
from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

from thesis_code.models.gans.hagan.hagan import save_mri


# NOTE: 3D alpha-GAN paper mentions a connective loss that is not used in the alpha-gan implementation benchmarked in KWON GAN???


class Discriminator(nn.Module):
    def __init__(self, channel=512, out_class=1, is_dis=True):
        super(Discriminator, self).__init__()
        self.is_dis = is_dis
        self.channel = channel
        n_class = out_class

        self.conv1 = nn.Conv3d(1, channel // 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(
            channel // 8, channel // 4, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm3d(channel // 4)
        self.conv3 = nn.Conv3d(
            channel // 4, channel // 2, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm3d(channel // 2)
        self.conv4 = nn.Conv3d(
            channel // 2, channel, kernel_size=4, stride=2, padding=1
        )
        self.bn4 = nn.BatchNorm3d(channel)

        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)

        output = F.sigmoid(h5.view(h5.size()[0], -1))

        return output


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, self.latent_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)

        output = h5.view(h5.size()[0], -1)

        return output


class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100):
        super(Code_Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(code_size, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l3 = nn.Linear(4096, 1)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = F.sigmoid(h3)

        return output


class Generator(nn.Module):
    def __init__(self, noise: int = 100):
        super(Generator, self).__init__()

        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(
            noise, 512, kernel_size=4, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm3d(512)

        self.tp_conv2 = nn.Conv3d(
            512, 256, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(256)

        self.tp_conv3 = nn.Conv3d(
            256, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(128)

        self.tp_conv4 = nn.Conv3d(
            128, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm3d(64)

        self.tp_conv5 = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, z):
        z = z.view(-1, self.noise, 1, 1, 1)
        h = self.tp_conv1(z)
        h = self.relu(self.bn1(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv5(h)

        h = F.tanh(h)

        return h


class LitAlphaGAN(L.LightningModule):
    def __init__(
        self,
        latent_dim: int = 1024,
        lambda_recon: float = 10.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(noise=latent_dim)
        self.discriminator = Discriminator()
        self.code_discriminator = Code_Discriminator(code_size=latent_dim)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.lambda_recon = lambda_recon
        self.automatic_optimization = False  # Manual optimization
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.automatic_optimization = False  # Manual optimization

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def sample_n(self, n) -> torch.Tensor:
        return self(self.sample_z(n))

    def forward(self, z):
        return self.generator(z)

    def discriminator_loss(self, real_data: Tensor) -> Tensor:
        fake_codes = self.encoder(real_data)
        fake_data = self.generator(fake_codes)  # From encoded data
        batch_size = real_data.size(0)
        latent_codes = self.sample_z(batch_size)
        latent_data = self.generator(latent_codes)  # From random distribution

        disc_real = self.discriminator(real_data)
        real_labels = torch.ones_like(disc_real)
        real_loss = F.binary_cross_entropy(disc_real, real_labels)

        disc_fake = self.discriminator(fake_data)
        fake_labels = torch.zeros_like(disc_fake)
        fake_loss = self.bce_loss(disc_fake, fake_labels)

        disc_latent = self.discriminator(latent_data)
        latent_labels = torch.zeros_like(disc_latent)
        latent_loss = self.bce_loss(disc_latent, latent_labels)

        return 2 * real_loss + fake_loss + latent_loss

    def code_discriminator_loss(self, real_data: Tensor) -> Tensor:
        batch_size = real_data.size(0)
        fake_codes = self.encoder(real_data)
        latent_codes = self.sample_z(batch_size)

        c_disc_latent = self.code_discriminator(latent_codes)
        latent_labels = torch.ones_like(c_disc_latent)
        latent_loss = self.bce_loss(c_disc_latent, latent_labels)

        c_disc_fake = self.code_discriminator(fake_codes)
        fake_labels = torch.zeros_like(c_disc_fake)
        fake_loss = self.bce_loss(c_disc_fake, fake_labels)

        return latent_loss + fake_loss

    def encoder_loss(self, real_data: Tensor) -> Tensor:
        fake_codes = self.encoder(real_data)
        fake_data = self.generator(fake_codes)  # From encoded data
        code_disc_latent = self.code_discriminator(fake_codes)
        latent_labels = torch.ones_like(code_disc_latent)

        recon_loss = self.l1_loss(real_data, fake_data)
        # Signs are flipped, check docs for BCELoss
        code_loss = self.bce_loss(code_disc_latent, latent_labels) - self.bce_loss(
            1 - code_disc_latent, 1 - latent_labels
        )
        return recon_loss + code_loss

    def generator_loss(self, real_data: Tensor) -> Tensor:
        fake_codes = self.encoder(real_data)
        fake_data = self.generator(fake_codes)  # From encoded data
        batch_size = real_data.size(0)
        latent_codes = self.sample_z(batch_size)
        latent_data = self.generator(latent_codes)  # From random distribution

        recon_loss = self.l1_loss(real_data, fake_data)

        disc_latent = self.discriminator(latent_data)
        latent_labels = torch.ones_like(disc_latent)
        # Signs are flipped, check docs for BCELoss
        adv_loss_latent = self.bce_loss(disc_latent, latent_labels) - self.bce_loss(
            1 - disc_latent, 1 - latent_labels
        )

        disc_fake = self.discriminator(fake_data)
        fake_labels = torch.ones_like(disc_fake)
        # Signs are flipped, check docs for BCELoss
        adv_loss_fake = self.bce_loss(disc_fake, fake_labels) - self.bce_loss(
            1 - disc_fake, 1 - fake_labels
        )

        return self.lambda_recon * recon_loss + adv_loss_latent + adv_loss_fake

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        real_data = batch

        # Optimizers
        opt_g, opt_d, opt_c, opt_e = self.optimizers()  # type: ignore

        # Encoder loss and optimization
        e_loss = self.encoder_loss(real_data=real_data)
        opt_e.zero_grad()
        self.manual_backward(e_loss)
        opt_e.step()

        # Generator loss and optimization
        g_loss = self.generator_loss(real_data=real_data)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        # Two because they do so - they offer no reason.
        opt_g.step()
        opt_g.step()

        # Discriminator loss and optimization
        d_loss = self.discriminator_loss(real_data=real_data)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # Code discriminator loss and optimization
        c_loss = self.code_discriminator_loss(real_data=real_data)
        opt_c.zero_grad()
        self.manual_backward(c_loss)
        opt_c.step()

        total_loss = (d_loss + g_loss + c_loss + e_loss).detach()

        # Log losses
        self.log_dict(
            {
                "d_loss": d_loss,
                "g_loss": g_loss,
                "c_loss": c_loss,
                "e_loss": e_loss,
                "total_loss": total_loss,
            }
        )

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        real_data = batch
        batch_size = real_data.size(0)

        # Enable gradient computation for gradient penalty calculation
        with torch.enable_grad():
            # Encoder loss and optimization
            e_loss = self.encoder_loss(real_data=real_data)

            # Generator loss and optimization
            g_loss = self.generator_loss(real_data=real_data)

            # Critic loss and optimization
            d_loss = self.discriminator_loss(real_data=real_data)

            # Code critic loss and optimization
            c_loss = self.code_discriminator_loss(real_data=real_data)

        if batch_idx == 0:
            # Save validation data
            log_dir = Path(self.logger.log_dir)  # type: ignore

            fake_images = self.sample_n(batch_size)
            synthetic_example_save_path = (
                log_dir / f"synthetic_example_{self.current_epoch}.nii.gz"
            )
            save_mri(fake_images, synthetic_example_save_path)

            # Save true data
            true_example_save_path = (
                log_dir / f"true_example_{self.current_epoch}.nii.gz"
            )
            save_mri(real_data, true_example_save_path)

        fake_data = self.generator(self.sample_z(batch_size))
        # Logging accuracy of discriminator with respect to cropped and small images simultaneously
        d_accuracy = (
            torch.mean(self.discriminator(real_data))
            + torch.mean(self.discriminator(fake_data))
        ) / 2

        # elapsed_time = time.time() - self.start_time

        # Log losses
        self.log_dict(
            {
                "val_d_loss": d_loss,
                "val_g_loss": g_loss,
                "val_c_loss": c_loss,
                "val_e_loss": e_loss,
                "val_total_loss": d_loss + g_loss + c_loss + e_loss,
                "val_d_accuracy": d_accuracy,
                # "elapsed_time": elapsed_time,
            },
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # Separate optimizers for generator, discriminator, and code discriminator
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        opt_c = torch.optim.Adam(
            self.code_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        opt_e = torch.optim.Adam(
            self.encoder.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        return [opt_g, opt_d, opt_c, opt_e]
