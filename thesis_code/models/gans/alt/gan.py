import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L


class GANModule(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        super(GANModule, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z):
        # For inference, typically just return the generator's output
        return self.generator(z)

    def sample(self, num_samples=1):
        # Generate samples using the generator
        z = torch.randn(
            num_samples, self.generator.latent_dim, device=self.generator.device
        )
        return self.generator(z)


class LitGAN(L.LightningModule):
    def __init__(self, generator, discriminator, latent_dim: int, device):
        super(LitGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.bce_loss = nn.BCELoss()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def sample_z(self, num_samples):
        return torch.randn(num_samples, self.latent_dim, device=self.device)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x_real = batch

        g_opt, d_opt = self.optimizers()  # type: ignore

        # discriminator loss
        z_d = self.sample_z(x_real.size(0))
        d_g_out = self.generator(z_d).detach()
        d_out_real = self.discriminator(x_real)
        d_out_fake = self.discriminator(d_g_out)

        d_loss = self.discriminator_loss(d_out_real, d_out_fake)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # generator loss
        z_g = self.sample_z(x_real.size(0))
        g_g_out = self.generator(z_g).detach()
        g_out_fake = self.discriminator(g_g_out)

        g_loss = self.generator_loss(g_out_fake)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)

    def discriminator_loss(
        self, d_real: torch.Tensor, d_fake: torch.Tensor
    ) -> torch.Tensor:
        # minimising BCE = maximising E[log(D(x)) + log(1 - D(G(z)))]
        #                = maximising E_real[log(D(x))] + E_fake[log(1 - D(G(z)))]

        # E_real[log(D(x))]
        d_real_loss = self.bce_loss(d_real, torch.ones_like(d_real))
        # E_fake[log(1 - D(G(z)))]
        d_fake_loss = self.discriminator_loss(d_fake, torch.zeros_like(d_fake))
        return d_real_loss + d_fake_loss

    def generator_loss(self, g_fake):
        # Could theoretically also use -self.bce_loss(g_fake, torch.zeros_like(g_fake))
        # For maximising E_fake[log(D(G(z)))] instead

        # minimising -bce = maximising bce = minimising E_fake[log(1 - D(G(z)))]
        return self.bce_loss(g_fake, torch.ones_like(g_fake))

    def configure_optimizers(self):
        g_optimizer = optim.Adam(self.generator.parameters())
        d_optimizer = optim.Adam(self.discriminator.parameters())
        return [g_optimizer, d_optimizer], []
