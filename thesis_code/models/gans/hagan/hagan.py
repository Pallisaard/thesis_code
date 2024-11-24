from pathlib import Path
import time
from typing import Dict, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import nibabel as nib

from thesis_code.models.gans.hagan.backbone.Model_HA_GAN_256 import (
    Generator,
    Discriminator,
    Encoder,
    Sub_Encoder,
    S_L,
    S_H,
)
from thesis_code.dataloading.transforms import normalize_to
from thesis_code.dataloading import MRISample  # , MRIDataset, MRIDataModule
# from thesis_code.metrics.ssi_3d import batch_ssi_3d


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
        self.G = Generator(latent_dim=self.latent_dim)
        self.D = Discriminator()
        self.E = Encoder()
        self.Sub_E = Sub_Encoder(latent_dim=self.latent_dim)
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lr_e = lr_e
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_time = time.time()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

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
        data_dict = prepare_data(batch=batch, latent_dim=self.latent_dim)
        real_images = data_dict["real_images"]
        batch_size = data_dict["batch_size"]
        real_images_small = data_dict["real_images_small"]
        crop_idx = data_dict["crop_idx"]
        real_images_crop = data_dict["real_images_crop"]
        noise = data_dict["noise"]
        real_labels = data_dict["real_labels"]
        fake_labels = data_dict["fake_labels"]

        d_opt, g_opt, e_opt, sub_e_opt = self.optimizers()  # type: ignore

        # D (D^H, D^L)
        d_opt.zero_grad()
        d_loss = compute_d_loss(
            D=self.D,
            G=self.G,
            bce_loss=self.bce_loss,
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
            noise=noise,
            real_labels=real_labels,
            fake_labels=fake_labels,
        )
        self.manual_backward(d_loss)
        d_opt.step()

        # G (G^A, G^H, G^L)
        g_opt.zero_grad()
        g_loss = compute_g_loss(
            G=self.G,
            D=self.D,
            bce_loss=self.bce_loss,
            noise=noise,
            crop_idx=crop_idx,
            real_labels=real_labels,
        )
        self.manual_backward(g_loss)
        g_opt.step()

        # E (E^H)
        e_opt.zero_grad()
        e_loss = compute_e_loss(
            E=self.E,
            G=self.G,
            l1_loss=self.l1_loss,
            real_images_crop=real_images_crop,
            lambda_1=self.lambda_1,
        )
        self.manual_backward(e_loss)
        e_opt.step()

        # Sub_E (E^G)
        self.Sub_E.zero_grad()
        sub_e_loss = compute_sub_e_loss(
            Sub_E=self.Sub_E,
            G=self.G,
            l1_loss=self.l1_loss,
            real_images=real_images,
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
            lambda_2=self.lambda_2,
        )
        self.manual_backward(sub_e_loss)
        sub_e_opt.step()

        self.log("d_loss", d_loss, logger=True, sync_dist=True)
        self.log("g_loss", g_loss, logger=True, sync_dist=True)
        self.log("e_loss", e_loss, logger=True, sync_dist=True)
        self.log("sub_e_loss", sub_e_loss, logger=True, sync_dist=True)

        total_loss = d_loss + g_loss + e_loss + sub_e_loss
        self.log("total_loss", total_loss, logger=True, sync_dist=True)

        # Log elapsed time
        elapsed_time = time.time() - self.start_time
        self.log("elapsed_time", elapsed_time, logger=True, sync_dist=True)

    def validation_step(self, batch: MRISample, batch_idx):
        data_dict = prepare_data(batch=batch, latent_dim=self.latent_dim)
        real_images = data_dict["real_images"]
        batch_size = data_dict["batch_size"]
        real_images_small = data_dict["real_images_small"]
        crop_idx = data_dict["crop_idx"]
        real_images_crop = data_dict["real_images_crop"]
        noise = data_dict["noise"]
        real_labels = data_dict["real_labels"]
        fake_labels = data_dict["fake_labels"]

        # Compute D loss
        d_loss = compute_d_loss(
            D=self.D,
            G=self.G,
            bce_loss=self.bce_loss,
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
            noise=noise,
            real_labels=real_labels,
            fake_labels=fake_labels,
        )

        # Compute G loss
        g_loss = compute_g_loss(
            G=self.G,
            D=self.D,
            bce_loss=self.bce_loss,
            noise=noise,
            crop_idx=crop_idx,
            real_labels=real_labels,
        )

        # Compute E loss
        e_loss = compute_e_loss(
            E=self.E,
            G=self.G,
            l1_loss=self.l1_loss,
            real_images_crop=real_images_crop,
            lambda_1=self.lambda_1,
        )

        # Compute Sub_E loss
        sub_e_loss = compute_sub_e_loss(
            Sub_E=self.Sub_E,
            G=self.G,
            l1_loss=self.l1_loss,
            real_images=real_images,
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
            lambda_2=self.lambda_2,
        )

        # Compute SSIM scores
        # ssim_score = batch_ssi_3d(real_images, fake_images, reduction="mean")

        if batch_idx == 0:
            # Save validation data
            fake_images = self.safe_sample(batch_size)
            sample_array = normalize_to(fake_images[0, 0].cpu().numpy(), -1, 1)
            sample_nii = numpy_to_nifti(sample_array)
            log_dir = Path(self.logger.log_dir)  # type: ignore
            file_path = (
                log_dir / f"validation_synthetic_example_{self.current_epoch}.nii.gz"
            )
            nib.save(sample_nii, file_path)  # type: ignore

            # Save true data
            true_array = normalize_to(real_images[0, 0].cpu().numpy(), -1, 1)
            true_nii = numpy_to_nifti(true_array)
            file_path = log_dir / f"validation_true_example_{self.current_epoch}.nii.gz"
            nib.save(true_nii, file_path)  # type: ignore

        # Log losses and SSIM scores
        self.log("val_d_loss", d_loss, logger=True, sync_dist=True)
        self.log("val_g_loss", g_loss, logger=True, sync_dist=True)
        self.log("val_e_loss", e_loss, logger=True, sync_dist=True)
        self.log("val_sub_e_loss", sub_e_loss, logger=True, sync_dist=True)

        total_loss = d_loss + g_loss + e_loss + sub_e_loss
        self.log("val_total_loss", total_loss, logger=True, sync_dist=True)

        # self.log("val_ssim_score", ssim_score, logger=True, sync_dist=True)

        # Log elapsed time
        elapsed_time = time.time() - self.start_time
        self.log("elapsed_time", elapsed_time, logger=True, sync_dist=True)

    def generate_from_noise(self, noise: torch.Tensor) -> torch.Tensor:
        out = self.G.generate(noise)
        return out

    def safe_sample(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn((num_samples, self.latent_dim), device=self.device)
        out_list = []
        for noise_slice in noise:
            out = self.generate_from_noise(noise_slice.unsqueeze(0))
            out_list.append(out)
        out = torch.cat(out_list, dim=0)
        return out

    def sample(self, num_samples: int) -> torch.Tensor:
        noise = torch.rand((num_samples, self.latent_dim), device=self.device)
        out = self.generate_from_noise(noise)
        return out

    def encode_to_small(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_crop_i_list = []
            for crop_idx_i in range(0, 256, 256 // 8):
                real_images_crop_i = S_H(x, crop_idx_i)
                encoded_crop_i = self.E(real_images_crop_i)
                encoded_crop_i_list.append(encoded_crop_i)
            encoded_crops = torch.cat(encoded_crop_i_list, dim=2).detach()
        return encoded_crops

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded_crops = self.encode_to_small(x)
        z_hat = self.Sub_E(encoded_crops)
        return z_hat


# Loss Functions
def compute_d_loss(
    D: nn.Module,
    G: nn.Module,
    bce_loss: nn.Module,
    real_images_crop: torch.Tensor,
    real_images_small: torch.Tensor,
    crop_idx: int,
    noise: torch.Tensor,
    real_labels: torch.Tensor,
    fake_labels: torch.Tensor,
) -> torch.Tensor:
    y_real_pred = D(real_images_crop, real_images_small, crop_idx)
    d_real_loss = bce_loss(y_real_pred, real_labels)
    fake_images, fake_images_small = G(noise, crop_idx=crop_idx)
    y_fake_pred = D(fake_images.detach(), fake_images_small.detach(), crop_idx)
    d_fake_loss = bce_loss(y_fake_pred, fake_labels)
    d_loss = d_real_loss + d_fake_loss
    return d_loss


def compute_g_loss(
    G: nn.Module,
    D: nn.Module,
    bce_loss: nn.Module,
    noise: torch.Tensor,
    crop_idx: int,
    real_labels: torch.Tensor,
) -> torch.Tensor:
    fake_images, fake_images_small = G(noise, crop_idx=crop_idx)
    y_fake_pred = D(fake_images, fake_images_small, crop_idx)
    g_loss = bce_loss(y_fake_pred, real_labels)
    return g_loss


def compute_e_loss(
    E: nn.Module,
    G: nn.Module,
    l1_loss: nn.Module,
    real_images_crop: torch.Tensor,
    lambda_1: float,
):
    encoded_crop = E(real_images_crop)
    x_hat = G.G_H(encoded_crop)
    e_loss = lambda_1 * l1_loss(x_hat, real_images_crop)
    return e_loss


def compute_sub_e_loss(
    E: nn.Module,
    Sub_E: nn.Module,
    G: nn.Module,
    l1_loss: nn.Module,
    real_images: torch.Tensor,
    real_images_crop: torch.Tensor,
    real_images_small: torch.Tensor,
    crop_idx: int,
    lambda_2: float,
):
    z_hat = Sub_E(E(real_images))
    sub_x_hat_rec, sub_x_hat_rec_small = G(z_hat, crop_idx=crop_idx)
    sub_e_loss = (
        lambda_2
        * (
            l1_loss(sub_x_hat_rec, real_images_crop)
            + l1_loss(sub_x_hat_rec_small, real_images_small)
        )
        / 2.0
    )
    return sub_e_loss


class DataDict(TypedDict):
    real_images: torch.Tensor
    batch_size: int
    real_images_small: torch.Tensor
    real_images_crop: torch.Tensor
    noise: torch.Tensor
    real_labels: torch.Tensor
    fake_labels: torch.Tensor
    crop_idx: int


# Data Preparation Function
def prepare_data(batch: MRISample, latent_dim: int) -> DataDict:
    real_images = batch["image"].float()
    batch_size = real_images.size(0)
    real_images_small = F.interpolate(real_images, scale_factor=0.25)
    crop_idx = np.random.randint(0, 256 * 7 // 8 + 1)
    real_images_crop = S_H(real_images, crop_idx)
    noise = torch.randn((batch_size, latent_dim), device=real_images.device)
    real_labels = torch.ones((batch_size, 1), device=real_images.device)
    fake_labels = torch.zeros((batch_size, 1), device=real_images.device)
    return {
        "real_images": real_images,
        "batch_size": batch_size,
        "real_images_small": real_images_small,
        "real_images_crop": real_images_crop,
        "noise": noise,
        "real_labels": real_labels,
        "fake_labels": fake_labels,
        "crop_idx": crop_idx,
    }


def numpy_to_nifti(array: np.ndarray) -> nib.Nifti1Image:  # type: ignore
    """
    Convert a 3D numpy array to a Nifti1Image with RAS orientation.

    Parameters:
    array (np.ndarray): 3D numpy array to convert.

    Returns:
    nib.Nifti1Image: Nifti image that can be saved as an .nii.gz file.
    """
    # Ensure the array is 3D
    if array.ndim != 3:
        raise ValueError("Input array must be 3D")

    # Create an identity affine matrix
    affine = np.eye(4)

    # Create the Nifti1Image
    nifti_img = nib.Nifti1Image(array, affine)  # type: ignore

    # Set the orientation to RAS
    nifti_img = nib.as_closest_canonical(nifti_img)  # type: ignore

    return nifti_img
