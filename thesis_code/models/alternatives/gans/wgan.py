import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch import autograd


class Discriminator(nn.Module):
    def __init__(self, channel=512):
        super(Discriminator, self).__init__()
        self.channel = channel
        n_class = 1

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

        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=2, padding=1)

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
        self.fc = nn.Linear(self.latent_dim, 512 * 4 * 4 * 4)
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

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))

        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv5(h)

        h = F.tanh(h)

        return h


def calc_gradient_penalty(model, real_data, fake_data, device="cuda", lambda_=10):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = interpolates.requires_grad_(True)

    discriminated_interpolates = model(interpolates)

    gradients = autograd.grad(
        outputs=discriminated_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(discriminated_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_


class LitWGAN(L.LightningModule):
    """WGAN without gradient policy."""

    def __init__(
        self,
        latent_dim: int,
        gradient_clamp: float = 0.01,
        n_critic_steps: int = 5,
    ):
        super().__init__()
        self.generator = Generator(latent_dim)
        self.critic = Discriminator()
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

    def sample_n(self, n) -> torch.Tensor:
        return self(self.sample_z(n))

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
