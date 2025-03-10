import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
)

from thesis_code.dataloading.utils import save_mri
from .backbone import CodeDiscriminator, Encoder, Generator, Discriminator


class LitKwonGan(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lambda_gp: float = 10.0,
        n_critic_steps: int = 5,
        lambda_recon: float = 10.0,
    ):
        super().__init__()
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters()

        self.generator = Generator(latent_dim)
        self.critic = Discriminator()
        self.code_critic = CodeDiscriminator(latent_dim)
        self.encoder = Encoder(latent_dim)
        self.lambda_gp = lambda_gp
        self.n_critic_steps = n_critic_steps
        self.lambda_recon = lambda_recon
        self.latent_dim = latent_dim

        self.ssim = StructuralSimilarityIndexMeasure(
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
        )

        self.automatic_optimization = False  # Manual optimization
        self.start_time = time.time()

    def verify_models(self, x):
        z = self.sample_z(x.size(0))
        assert self.generator(z).shape == x.shape
        assert self.critic(x).shape == (x.shape[0], 1)
        assert self.code_critic(z).shape == (z.shape[0], 1)
        assert self.encoder(x).shape == z.shape

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def sample_1(self) -> torch.Tensor:
        return self(self.sample_z(1))[0]

    def sample_n(self, n) -> torch.Tensor:
        return self(self.sample_z(n))

    def sample(self, n) -> torch.Tensor:
        return self.sample_n(n)

    def forward(self, z):
        return self.generator(z)

    def reconstruction_loss(self, real_data: torch.Tensor, recon_data: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(real_data, recon_data)

    def generator_loss(
        self,
        real_data: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_data.size(0)
        random_codes = self.sample_z(batch_size)
        latent_codes = self.encoder(real_data)
        fake_data = self.generator(random_codes)
        recon_data = self.generator(latent_codes)
        recon_critic_score = self.critic(recon_data)
        fake_critic_score = self.critic(fake_data)

        return (
            self.lambda_recon * self.reconstruction_loss(real_data, recon_data)
            - torch.mean(fake_critic_score)
            - torch.mean(recon_critic_score)
        )

    def encoder_loss(
        self,
        real_data: torch.Tensor,
    ) -> torch.Tensor:
        latent_codes = self.encoder(real_data)
        recon_code_critic_score = self.code_critic(latent_codes)
        recon_data = self.generator(latent_codes)

        return -torch.mean(recon_code_critic_score) + self.lambda_recon * self.reconstruction_loss(
            real_data, recon_data
        )

    def generator_encoder_loss(self, real_data: torch.Tensor):
        batch_size = real_data.size(0)
        random_codes = self.sample_z(batch_size)
        latent_codes = self.encoder(real_data)
        fake_data = self.generator(random_codes)
        recon_data = self.generator(latent_codes)
        recon_critic_score = self.critic(recon_data)
        fake_critic_score = self.critic(fake_data)
        recon_code_critic_score = self.code_critic(latent_codes)

        c_loss = -torch.mean(recon_code_critic_score)
        d_loss = -torch.mean(fake_critic_score) - torch.mean(recon_critic_score)
        l1_loss = self.lambda_recon * self.reconstruction_loss(real_data, recon_data)
        return l1_loss + c_loss + d_loss

    def critic_loss(
        self,
        real_data: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_data.size(0)
        random_codes = self.sample_z(batch_size)

        latent_codes = self.encoder(real_data)

        fake_data = self.generator(random_codes)
        recon_data = self.generator(latent_codes)

        real_critic_score = self.critic(real_data)
        fake_critic_score = self.critic(fake_data)
        recon_critic_score = self.critic(recon_data)

        gp_loss_fake = calc_gradient_penalty(self.critic, real_data, fake_data)
        gp_loss_recon = calc_gradient_penalty(self.critic, real_data, recon_data)

        return (
            torch.mean(fake_critic_score)
            + torch.mean(recon_critic_score)
            - 2.0 * torch.mean(real_critic_score)
            + self.lambda_gp * gp_loss_fake
            + self.lambda_gp * gp_loss_recon
        )

    def code_critic_loss(
        self,
        real_data: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_data.size(0)
        random_codes = self.sample_z(batch_size)
        latent_codes = self.encoder(real_data)

        real_code_critic_score = self.code_critic(random_codes)
        fake_code_critic_score = self.code_critic(latent_codes)

        gp_loss = calc_gradient_penalty(self.code_critic, random_codes, latent_codes)

        return torch.mean(fake_code_critic_score) - torch.mean(real_code_critic_score) + self.lambda_gp * gp_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # with torch.autograd.set_detect_anomaly(True):
        # Optimizers
        opt_g, opt_d, opt_c, opt_e = self.optimizers()  # type: ignore

        real_data = batch

        # Encoder loss and optimization
        e_loss = self.encoder_loss(real_data=real_data)
        opt_e.zero_grad()
        self.manual_backward(e_loss)
        opt_e.step()

        # Generator loss and optimization
        g_loss = self.generator_loss(real_data=real_data)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.step()

        # opt_g.zero_grad()
        # opt_e.zero_grad()
        # e_g_loss = self.generator_encoder_loss(real_data=real_data)
        # self.manual_backward(e_g_loss)
        # opt_e.step()
        # opt_g.step()
        # opt_g.step()

        # critic loss and optimization
        d_loss = self.critic_loss(real_data=real_data)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # Code critic loss and optimization
        c_loss = self.code_critic_loss(real_data=real_data)
        opt_c.zero_grad()
        self.manual_backward(c_loss)
        opt_c.step()

        # total_loss = d_loss + g_loss + c_loss + e_loss
        total_loss = d_loss + c_loss + e_loss + g_loss
        elapsed_time = time.time() - self.start_time
        # Log losses
        self.log_dict(
            {
                "d_loss": d_loss,
                "g_loss": g_loss,
                "c_loss": c_loss,
                "e_loss": e_loss,
                # "e_g_loss": e_g_loss,
                "total_loss": total_loss,
                "elapsed_time": elapsed_time,
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
            d_loss = self.critic_loss(real_data=real_data)

            # Code critic loss and optimization
            c_loss = self.code_critic_loss(real_data=real_data)

        if batch_idx == 0:
            # Save validation data
            log_dir = Path(self.logger.log_dir)  # type: ignore

            fake_images = self.sample_n(batch_size)
            synthetic_example_save_path = log_dir / f"synthetic_example_{self.current_epoch}.nii.gz"
            save_mri(fake_images, synthetic_example_save_path, ignore_intensity_variation=True)

            # Save true data
            true_example_save_path = log_dir / f"true_example_{self.current_epoch}.nii.gz"
            save_mri(real_data, true_example_save_path, ignore_intensity_variation=True)

        fake_data = self.generator(self.sample_z(batch_size))
        # Logging accuracy of discriminator with respect to cropped and small images simultaneously
        d_accuracy = (torch.mean(self.critic(real_data)) + torch.mean(self.critic(fake_data))) / 2

        elapsed_time = time.time() - self.start_time

        # Log losses
        self.log_dict(
            {
                "val_d_loss": d_loss,
                "val_g_loss": g_loss,
                "val_c_loss": c_loss,
                "val_e_loss": e_loss,
                "val_total_loss": d_loss + g_loss + c_loss + e_loss,
                "val_d_accuracy": d_accuracy,
                "elapsed_time": elapsed_time,
            },
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # Separate optimizers for generator, critic, and code critic
        opt_g = torch.optim.Adam(self.generator.parameters())
        opt_d = torch.optim.Adam(self.critic.parameters())
        opt_c = torch.optim.Adam(self.code_critic.parameters())
        opt_e = torch.optim.Adam(self.encoder.parameters())
        return [opt_g, opt_d, opt_c, opt_e]


def calc_gradient_penalty(
    model: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: str = "cuda",
    lambda_: float = 10.0,
):
    alpha = torch.rand(real_data.size(0), *([1] * len(real_data.shape[1:])))
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = interpolates.requires_grad_(True)

    discriminated_interpolates = model(interpolates)

    gradients = torch.autograd.grad(
        outputs=discriminated_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(discriminated_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    # Norm with added epsilon to avoid division by zero
    gradient_norms = ((gradients * gradients + 1e-12).sum(-1)).sqrt()
    # Compute the gradient penalty

    return lambda_ * ((gradient_norms - 1) ** 2).mean()
