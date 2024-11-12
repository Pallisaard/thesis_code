from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from thesis_code.models.gans.hagan.backbone.Model_HA_GAN_256 import (
    Generator,
    Discriminator,
    Encoder,
    Sub_Encoder,
    S_L,
    S_H,
)
from thesis_code.dataloading import MRISample, MRIDataset, MRIDataModule


class HAGAN(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        lr_g: float = 0.0001,
        lr_d: float = 0.0004,
        lr_e: float = 0.0001,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.G = Generator(mode="train", latent_dim=self.latent_dim)
        self.D = Discriminator()
        self.E = Encoder()
        self.Sub_E = Sub_Encoder(latent_dim=self.latent_dim)
        self.S
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lr_e = lr_e

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_l1 = nn.L1Loss()

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

    def training_step(self, batch: MRISample, batch_idx, optimizer_idx):
        real_images = batch["image"]
        batch_size = real_images.size(0)
        real_images = real_images.float()
        real_images_small = F.interpolate(real_images, scale_factor=0.25)
        crop_idx = np.random.randint(0, 256 * 7 // 8 + 1)
        # real_images_crop = real_images[:, :, crop_idx : crop_idx + 256 // 8, :, :]
        real_images_crop = S_H(real_images, crop_idx)

        self.real_labels = torch.ones((batch_size, 1), device=real_images.device)
        self.fake_labels = torch.zeros((batch_size, 1), device=real_images.device)

        if optimizer_idx == 0:  # D
            self.D.zero_grad()
            self.G.requires_grad_(False)
            self.E.requires_grad_(False)
            self.Sub_E.requires_grad_(False)

            y_real_pred = self.D(real_images_crop, real_images_small, crop_idx)
            d_real_loss = self.loss_bce(y_real_pred, self.real_labels)
            noise = torch.randn(
                (batch_size, self.latent_dim),
                device=real_images.device,
            )
            fake_images, fake_images_small = self.G(noise, crop_idx=crop_idx)
            y_fake_pred = self.D(fake_images, fake_images_small, crop_idx)

            d_fake_loss = self.loss_bce(y_fake_pred, self.fake_labels)
            d_loss = d_real_loss + d_fake_loss
            self.log(
                "d_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return d_loss

        elif optimizer_idx == 1:  # G
            self.G.zero_grad()
            self.D.requires_grad_(False)
            self.E.requires_grad_(False)
            self.Sub_E.requires_grad_(False)

            noise = torch.randn(
                (batch_size, self.latent_dim), device=real_images.device
            )
            fake_images, fake_images_small = self.G(
                noise, crop_idx=crop_idx, class_label=None
            )
            y_fake_g = self.D(fake_images, fake_images_small, crop_idx)
            g_loss = self.loss_bce(y_fake_g, self.real_labels)

            self.log(
                "g_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return g_loss

        elif optimizer_idx == 2:  # E
            self.E.zero_grad()
            self.G.requires_grad_(False)
            self.D.requires_grad_(False)
            self.Sub_E.requires_grad_(False)

            z_hat = self.E(real_images_crop)
            x_hat = self.G(z_hat, crop_idx=None)
            e_loss = self.loss_l1(x_hat, real_images_crop)
            self.log(
                "e_loss",
                e_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return e_loss

        elif optimizer_idx == 3:  # Sub_E
            self.Sub_E.zero_grad()
            self.G.requires_grad_(False)
            self.D.requires_grad_(False)
            self.E.requires_grad_(False)

            with torch.no_grad():
                z_hat_i_list = []
                for crop_idx_i in range(0, 256, 256 // 8):
                    real_images_crop_i = S_H(real_images, crop_idx_i)
                    z_hat_i = self.E(real_images_crop_i)
                    z_hat_i_list.append(z_hat_i)
                z_hat = torch.cat(z_hat_i_list, dim=2).detach()
            sub_z_hat = self.Sub_E(z_hat)
            sub_x_hat_rec, sub_x_hat_rec_small = self.G(sub_z_hat, crop_idx=crop_idx)

            sub_e_loss = (
                self.loss_l1(sub_x_hat_rec, real_images_crop)
                + self.loss_l1(sub_x_hat_rec_small, real_images_small)
            ) / 2.0
            self.log(
                "sub_e_loss",
                sub_e_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return sub_e_loss

    def sample(self, num_samples: int):
        noise = torch.randn((num_samples, self.latent_dim), device=self.device)
        return self.G(noise)

    def encode(self, x: torch.Tensor):
        return self.E(x)
