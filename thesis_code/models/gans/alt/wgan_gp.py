import torch
import torch.nn as nn
import pytorch_lightning as L


class LitWGANGP(L.LightningModule):
    """WGAN with gradient policy."""

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int,
        gp_weight: float = 10.0,
        n_critic_steps: int = 5,
    ):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.n_critic_steps = n_critic_steps
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

    def forward(self, z):
        return self.generator(z)

    def critic_gradient_policy(
        self, real_data: torch.Tensor, fake_data: torch.Tensor, gp_weight: float = 10.0
    ) -> torch.Tensor:
        interpolates_grad = self._interpolate_data_with_gradient(real_data, fake_data)

        batch_size = fake_data.size(0)
        # Computes the gradient of the critic with respect to the fake data
        gradient = torch.autograd.grad(
            outputs=self.critic(interpolates_grad).sum(),
            inputs=interpolates_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Flatten so the norm computation is easier
        gradient = gradient.view(batch_size, -1)
        # Norm with added epsilon to avoid division by zero
        gradient_norms = torch.sqrt(torch.sum(gradient**2, dim=1) + 1e-12)
        # Compute the gradient penalty

        return gp_weight * torch.mean((gradient_norms - 1) ** 2)

    def critic_loss(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        # negative sign because we want to maximize the critic loss
        critic_real_loss = self.critic(real_data).mean()
        critic_fake_loss = self.critic(fake_data).mean()
        gp_loss = self.critic_gradient_policy(real_data, fake_data)

        # Equivalent to: return -(critic_real_loss - critic_fake_loss) + gp_loss
        return critic_fake_loss - critic_real_loss + gp_loss

    def generator_loss(self, fake_data: torch.Tensor) -> torch.Tensor:
        # We want to maximize the critic loss on fake data, so we minimize the negative of it
        return -self.critic(fake_data).mean()

    def training_step(self, batch: torch.Tensor, batch_idx):
        real_data = batch
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

        # Train generator
        self.generator.zero_grad()
        z = self.sample_z(batch_size)
        fake_data = self.generator(z)
        generator_loss = self.generator_loss(fake_data)
        generator_loss.backward()
        self.generator_opt.step()

        return {"critic_loss": critic_loss, "generator_loss": generator_loss}

    def _interpolate_data_with_gradient(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.rand(1).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates_grad = torch.autograd.Variable(interpolates, requires_grad=True)
        return interpolates_grad


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view((batch_size, *self.shape))


if __name__ == "__main__":
    critic = nn.Sequential(nn.Flatten(), nn.Linear(27, 10), nn.ReLU(), nn.Linear(10, 1))
    generator = nn.Sequential(nn.Linear(3, 27), nn.ReLU(), Reshape((1, 3, 3, 3)))

    wgan_gp = LitWGANGP(generator, critic, latent_dim=3)
    wgan_gp.verify_models(torch.randn(8, 1, 3, 3, 3))
    gp_loss = wgan_gp.critic_gradient_policy(
        real_data=torch.randn(2, 1, 3, 3, 3), fake_data=torch.randn(2, 1, 3, 3, 3)
    )
    print(gp_loss)
