import time
from pathlib import Path

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from thesis_code.models.gans.hagan.hagan import save_mri


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        n_class = 1

        self.conv1 = nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, n_class, kernel_size=4, stride=2, padding=1)

    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        output = h5

        return output


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 1024):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)
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
        z = z.view(-1, self.latent_dim)
        h = self.fc(z)
        h = h.view(-1, 512, 4, 4, 4)
        h = F.relu(self.bn1(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv5(h)

        h = F.tanh(h)

        return h


class LitWGANGP(L.LightningModule):
    """WGAN with gradient policy."""

    def __init__(
        self,
        latent_dim: int,
        gp_weight: float = 10.0,
        n_critic_steps: int = 1,
        n_generator_steps: int = 5,
    ):
        super().__init__()
        self.generator = Generator(latent_dim=latent_dim)
        self.critic = Critic()
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.n_critic_steps = n_critic_steps
        self.n_generator_steps = n_generator_steps
        self.automatic_optimization = False  # Manual optimization

    def verify_models(self, x):
        z = self.sample_z(x.size(0))
        assert (
            self.generator(z).shape == x.shape
        ), f"{self.generator(z).shape} != {x.shape}"
        assert self.critic(x).shape == (
            x.shape[0],
            1,
        ), f"{self.critic(x).shape} != {(x.shape[0], 1)}"

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def sample_n(self, n) -> torch.Tensor:
        return self(self.sample_z(n))

    def forward(self, z):
        return self.generator(z)

    def critic_loss(self, real_data: torch.Tensor) -> torch.Tensor:
        batch_size = real_data.size(0)
        z = self.sample_z(batch_size)
        fake_data = self.generator(z)

        # negative sign because we want to maximize the critic loss
        critic_real_loss = -self.critic(real_data).mean()
        critic_fake_loss = self.critic(fake_data).mean()
        gp_loss = calc_gradient_penalty(
            self.critic, real_data, fake_data, lambda_=self.gp_weight
        )

        # Equivalent to: return -(critic_real_loss - critic_fake_loss) + gp_loss
        return critic_fake_loss + critic_real_loss + gp_loss

    def generator_loss(self, real_data: torch.Tensor) -> torch.Tensor:
        batch_size = real_data.size(0)
        z = self.sample_z(batch_size)
        fake_data = self.generator(z)
        # We want to maximize the critic loss on fake data, so we minimize the negative of it
        return -self.critic(fake_data).mean()

    def training_step(self, batch: torch.Tensor, batch_idx):
        c_opt, g_opt = self.optimizers()  # type: ignore

        real_data = batch

        # Train critic
        critics_mean_loss: list[torch.Tensor] = []

        for _ in range(self.n_critic_steps):
            self.critic.zero_grad()
            critic_loss = self.critic_loss(real_data=real_data)
            critic_loss.backward()
            c_opt.step()
            critics_mean_loss.append(critic_loss.detach())

        # Train generator
        generator_mean_loss: list[torch.Tensor] = []

        for _ in range(self.n_generator_steps):
            self.generator.zero_grad()
            generator_loss = self.generator_loss(real_data=real_data)
            generator_loss.backward()
            g_opt.step()
            generator_mean_loss.append(generator_loss.detach())

        return {
            "critic_loss": torch.mean(torch.stack(critics_mean_loss)),
            "generator_loss": torch.mean(torch.stack(generator_mean_loss)),
        }

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        real_data = batch
        batch_size = real_data.size(0)

        # Enable gradient computation for gradient penalty calculation
        with torch.enable_grad():
            # Encoder loss and optimization
            g_loss = self.generator_loss(real_data=real_data)

            # Critic loss and optimization
            c_loss = self.critic_loss(real_data=real_data)

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
            torch.mean(self.critic(real_data)) + torch.mean(self.critic(fake_data))
        ) / 2

        # elapsed_time = time.time() - self.start_time

        # Log losses
        self.log_dict(
            {
                "val_c_loss": c_loss,
                "val_g_loss": g_loss,
                "val_total_loss": c_loss + g_loss,
                "val_d_accuracy": d_accuracy,
                # "elapsed_time": elapsed_time,
            },
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # Separate optimizers for generator, critic, and code critic
        opt_g = torch.optim.Adam(self.generator.parameters())
        opt_c = torch.optim.Adam(self.critic.parameters())
        return [opt_g, opt_c]


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

    return ((gradient_norms - 1) ** 2).mean()
