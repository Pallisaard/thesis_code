import torch
import torch.nn as nn
import pytorch_lightning as L

from thesis_code.dataloading.mri_sample import MRISample


class LitWGAN(L.LightningModule):
    """WGAN without gradient policy."""

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int,
        gradient_clamp: float = 0.01,
        n_critic_steps: int = 5,
    ):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.gradient_clamp = gradient_clamp
        self.n_critic_steps = n_critic_steps
        self.automatic_optimization = False  # Manual optimization

    def verify_models(self, x):
        z = self.sample_z(x.size(0))
        assert self.generator(z).shape == x.shape
        assert self.critic(x).shape == (x.shape[0], 1)

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def forward(self, z):
        return self.generator(z)

    def critic_loss(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        # negative sign because we want to maximize the critic loss
        return -(self.critic(real_data).mean() - self.critic(fake_data).mean())

    def generator_loss(self, fake_data: torch.Tensor) -> torch.Tensor:
        # We want to maximize the critic loss on fake data, so we minimize the negative of it
        return -self.critic(fake_data).mean()

    def training_step(self, batch: MRISample, batch_idx):
        real_data = batch["image"]
        batch_size = real_data.size(0)

        critic_loss = None

        # Train critic
        for _ in range(self.n_critic_steps):
            self.critic.zero_grad()
            z = self.sample_z(batch_size)
            fake_data = self.generator(z)
            critic_loss = self.critic_loss(real_data, fake_data)
            critic_loss.backward()
            self.critic_opt.step()

            for p in self.critic.parameters():
                p.data.clamp_(-self.gradient_clamp, self.gradient_clamp)

        # Train generator
        self.generator.zero_grad()
        z = self.sample_z(batch_size)
        fake_data = self.generator(z)
        generator_loss = self.generator_loss(fake_data)
        generator_loss.backward()
        self.generator_opt.step()

        return {"critic_loss": critic_loss, "generator_loss": generator_loss}
