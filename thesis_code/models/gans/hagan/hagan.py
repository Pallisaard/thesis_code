from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from thesis_code.models.gans.hagan.backbone.Model_HA_GAN_256 import (
    Generator,
    Discriminator,
    Encoder,
    Sub_Encoder,
    S_L,
    S_H,
)
from thesis_code.metrics.ssi_3d import batch_ssi_3d
from thesis_code.dataloading import MRISample, MRIDataset, MRIDataModule


class HAGAN(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lr_g: float = 0.0001,
        lr_d: float = 0.0004,
        lr_e: float = 0.0001,
        # Does not show up in original code, but is mentioned in the paper
        lambda_1: float = 5.0,
        lambda_2: float = 5.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.G = Generator(mode="train", latent_dim=self.latent_dim)
        self.D = Discriminator()
        self.E = Encoder()
        self.Sub_E = Sub_Encoder(latent_dim=self.latent_dim)
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lr_e = lr_e
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_l1 = nn.L1Loss()

        self.automatic_optimization = False

    def configure_optimizers(self):
        g_optimizer = optim.Adam(
            self.G.parameters(), lr=self.lr_g, betas=(0.0, 0.999), eps=1e-8
        )
        d_optimizer = optim.Adam(
            self.D.parameters(), lr=self.lr_d, betas=(0.0, 0.999), eps=1e-8
        )
        e_optimizer = optim.Adam(
            self.E.parameters(), lr=self.lr_e, betas=(0.0, 0.999), eps=1e-8
        )
        sub_e_optimizer = optim.Adam(
            self.Sub_E.parameters(), lr=self.lr_e, betas=(0.0, 0.999), eps=1e-8
        )
        return [
            d_optimizer,
            g_optimizer,
            e_optimizer,
            sub_e_optimizer,
        ]

    def training_step(self, batch: MRISample, batch_idx):
        real_images = batch["image"]
        batch_size = real_images.size(0)
        real_images = real_images.float()
        real_images_small = F.interpolate(real_images, scale_factor=0.25)
        crop_idx = np.random.randint(0, 256 * 7 // 8 + 1)
        real_images_crop = S_H(real_images, crop_idx)
        noise = torch.randn((batch_size, self.latent_dim), device=real_images.device)

        self.real_labels = torch.ones((batch_size, 1), device=real_images.device)
        self.fake_labels = torch.zeros((batch_size, 1), device=real_images.device)

        d_opt, g_opt, e_opt, sub_e_opt = self.optimizers()  # type: ignore

        # D (D^H, D^L)
        self.D.requires_grad_(True)
        self.G.requires_grad_(False)
        self.E.requires_grad_(False)
        self.Sub_E.requires_grad_(False)
        self.D.zero_grad()
        d_loss = self.compute_d_loss(
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
            noise=noise,
        )
        self.manual_backward(d_loss)
        d_opt.step()

        # G (G^A, G^H, G^L)
        self.D.requires_grad_(False)
        self.G.requires_grad_(True)
        self.E.requires_grad_(False)
        self.Sub_E.requires_grad_(False)
        self.G.zero_grad()
        g_loss = self.compute_g_loss(
            noise=noise,
            crop_idx=crop_idx,
        )
        self.manual_backward(g_loss)
        g_opt.step()

        # E (E^H)
        self.G.requires_grad_(False)
        self.D.requires_grad_(False)
        self.e.requires_grad_(True)
        self.Sub_E.requires_grad_(False)
        self.E.zero_grad()
        e_loss = self.compute_e_loss(real_images_crop=real_images_crop)
        self.manual_backward(e_loss)
        e_opt.step()

        # Sub_E (E^G)
        self.G.requires_grad_(False)
        self.D.requires_grad_(False)
        self.E.requires_grad_(False)
        self.Sub_E.requires_grad_(True)
        self.Sub_E.zero_grad()
        sub_e_loss = self.compute_sub_e_loss(
            real_images=real_images,
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
        )
        self.manual_backward(sub_e_loss)
        sub_e_opt.step()

        self.log("d_loss", d_loss, on_step=True, on_epoch=True, logger=True)
        self.log("g_loss", g_loss, on_step=True, on_epoch=True, logger=True)
        self.log("e_loss", e_loss, on_step=True, on_epoch=True, logger=True)
        self.log("sub_e_loss", sub_e_loss, on_step=True, on_epoch=True, logger=True)

    def validation_step(self, batch: MRISample, batch_idx):
        real_images = batch["image"]
        batch_size = real_images.size(0)
        real_images = real_images.float()
        real_images_small = F.interpolate(real_images, scale_factor=0.25)
        crop_idx = np.random.randint(0, 256 * 7 // 8 + 1)
        real_images_crop = S_H(real_images, crop_idx)
        noise = torch.randn((batch_size, self.latent_dim), device=real_images.device)
        fake_images, fake_images_small = self.G(noise, crop_idx=crop_idx)

        self.real_labels = torch.ones((batch_size, 1), device=real_images.device)
        self.fake_labels = torch.zeros((batch_size, 1), device=real_images.device)

        # Compute D loss
        d_loss = self.compute_d_loss(
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
            noise=noise,
        )

        # Compute G loss
        g_loss = self.compute_g_loss(noise=noise, crop_idx=crop_idx)

        # Compute E loss
        e_loss = self.compute_e_loss(real_images_crop=real_images_crop)

        # Compute Sub_E loss
        sub_e_loss = self.compute_sub_e_loss(
            real_images=real_images,
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
        )

        # Compute SSIM scores
        ssim_score = batch_ssi_3d(real_images, fake_images)

        # Log losses and SSIM scores
        self.log("val_d_loss", d_loss, logger=True)
        self.log("val_g_loss", g_loss, logger=True)
        self.log("val_e_loss", e_loss, logger=True)
        self.log("val_sub_e_loss", sub_e_loss, logger=True)
        self.log("val_ssim_score", ssim_score, logger=True)

        return {
            "val_d_loss": d_loss,
            "val_g_loss": g_loss,
            "val_e_loss": e_loss,
            "val_sub_e_loss": sub_e_loss,
            "val_ssim_score": ssim_score,
        }

    def compute_d_loss(
        self,
        real_images_crop: torch.Tensor,
        real_images_small: torch.Tensor,
        crop_idx: Optional[int],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the discriminator loss.
        Computes
            Max_{D^H} [log(D^H(S^H(x^H,r),r)) + log(1 - D^H(G^H(z, r)))]$
        and
            Max_{D^L} [log(D^L(x^L)) + log(1 - D^L(G^L(z)))]
        simultaneously.

        """
        y_real_pred = self.D(real_images_crop, real_images_small, crop_idx)
        d_real_loss = self.loss_bce(y_real_pred, self.real_labels)
        fake_images, fake_images_small = self.G(noise, crop_idx=crop_idx)
        y_fake_pred = self.D(fake_images, fake_images_small, crop_idx)
        d_fake_loss = self.loss_bce(y_fake_pred, self.fake_labels)
        d_loss = d_real_loss + d_fake_loss
        return d_loss

    def compute_g_loss(
        self,
        noise: torch.Tensor,
        crop_idx: Optional[int],
    ) -> torch.Tensor:
        """Compute the generator loss.

        Computes
            Min_{G^H} Min_{G^A} [log(D^H(G^H(S^L(G^A(z), r))))]
        and
            Min_{G^L} Min_{G^A}[log(D^L(G^L(G^A(z))))]
        simultaneously.
        """
        fake_images, fake_images_small = self.G(noise, crop_idx=crop_idx)
        y_fake_pred = self.D(fake_images, fake_images_small, crop_idx)
        g_loss = self.loss_bce(y_fake_pred, self.real_labels)
        return g_loss

    def compute_e_loss(self, real_images_crop):
        """Compute the encoder loss.

        computes
            Min_{E^H} lambda_1 L1(X^H - G^H(E^H(x)))
        """
        encoded_crop = self.E(real_images_crop)
        x_hat = self.G.G_H(encoded_crop)
        e_loss = self.lambda_1 * self.loss_l1(x_hat, real_images_crop)
        return e_loss

    def compute_sub_e_loss(
        self, real_images, real_images_crop, real_images_small, crop_idx
    ):
        """Compute the sub-encoder loss.

        computes
            Min_{E^G} lambda_2 [L1(X^L - G^L(G^A(z))) + L1(S^H(X^H, r) - G^H(S^L(G^A(z), r)))] / 2.0
        note that in the original code, the L1 loss is divided by 2.0, which wasn't mentioned in the paper. Likewise, lambda_2 = 5 was mentioned in the paper but wasn't implemented.
        """
        z_hat = self.encode(real_images)
        sub_x_hat_rec, sub_x_hat_rec_small = self.G(z_hat, crop_idx=crop_idx)
        sub_e_loss = (
            self.lambda_2
            * (
                self.loss_l1(sub_x_hat_rec, real_images_crop)
                + self.loss_l1(sub_x_hat_rec_small, real_images_small)
            )
            / 2.0
        )
        return sub_e_loss

    def generate_from_noise(self, noise: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.G.generate(noise)
        return out

    def sample(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, self.latent_dim), device=self.device)
        with torch.no_grad():
            out = self.generate_from_noise(noise)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_crop_i_list = []
            for crop_idx_i in range(0, 256, 256 // 8):
                real_images_crop_i = S_H(x, crop_idx_i)
                encoded_crop_i = self.E(real_images_crop_i)
                encoded_crop_i_list.append(encoded_crop_i)
            encoded_crops = torch.cat(encoded_crop_i_list, dim=2).detach()
        z_hat = self.Sub_E(encoded_crops)
        return z_hat
