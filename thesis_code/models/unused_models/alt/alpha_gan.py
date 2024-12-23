import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as L
import torch.nn.functional as F


class LitAlphaGAN(L.LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        code_discriminator: nn.Module,
        encoder: nn.Module,
        lambda_recon: float = 1.0,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.code_discriminator = code_discriminator
        self.encoder = encoder
        self.lambda_recon = lambda_recon
        self.automatic_optimization = False  # Manual optimization

    def forward(self, x):
        return self.generator(x)

    def discriminator_loss(
        self,
        real_data: Tensor,
        fake_data: Tensor,
        real_codes: Tensor,
    ) -> Tensor:
        d_real = self.discriminator(real_data)
        real_labels = torch.ones_like(d_real)
        real_loss = F.binary_cross_entropy(d_real, real_labels)

        d_fake = self.discriminator(fake_data)
        fake_labels = torch.zeros_like(d_fake)
        fake_loss = F.binary_cross_entropy(d_fake, fake_labels)

        d_real_codes = self.discriminator(real_codes)
        real_latent_labels = torch.zeros_like(d_real_codes)
        latent_loss = F.binary_cross_entropy(
            self.discriminator(real_codes), real_latent_labels
        )

        return real_loss + fake_loss + latent_loss

    def code_discriminator_loss(self, real_codes: Tensor, fake_codes: Tensor) -> Tensor:
        fake_labels = torch.zeros_like(self.code_discriminator(fake_codes))
        fake_loss = F.binary_cross_entropy(
            self.code_discriminator(fake_codes), fake_labels
        )

        real_labels = torch.ones_like(self.code_discriminator(real_codes))
        real_loss = F.binary_cross_entropy(
            self.code_discriminator(real_codes), real_labels
        )

        return real_loss + fake_loss

    def reconstruction_loss(self, real_data: Tensor, fake_data_recon: Tensor) -> Tensor:
        return self.lambda_recon * F.l1_loss(real_data, fake_data_recon)

    def encoder_loss(
        self,
        fake_codes: Tensor,
    ) -> Tensor:
        code_disc_latent = self.code_discriminator(fake_codes)
        latent_labels = torch.ones_like(code_disc_latent)
        code_loss = -F.binary_cross_entropy(code_disc_latent, latent_labels)
        code_loss += F.binary_cross_entropy(1 - code_disc_latent, 1 - latent_labels)
        return code_loss

    def generator_loss(self, fake_data: Tensor, fake_data_recon: Tensor) -> Tensor:
        # Adversarial loss for samples generated from random noise
        disciminator_fake = self.discriminator(fake_data)
        fake_labels = torch.ones_like(disciminator_fake)
        adv_loss_fake = -F.binary_cross_entropy(disciminator_fake, fake_labels)
        adv_loss_fake += F.binary_cross_entropy(1 - disciminator_fake, 1 - fake_labels)

        # Adversarial loss for reconstructed samples
        # Better stability with this formula,
        disc_recon = self.discriminator(fake_data_recon)
        recon_labels = torch.ones_like(disc_recon)
        adv_loss_recon = -F.binary_cross_entropy(disc_recon, recon_labels)
        adv_loss_recon += F.binary_cross_entropy(1 - disc_recon, 1 - recon_labels)

        return adv_loss_fake + adv_loss_recon

    def training_step(self, batch, batch_idx):
        real_data, _ = batch
        batch_size = real_data.size(0)
        device = real_data.device

        # Optimizers
        opt_g, opt_d, opt_c, opt_e = self.optimizers()  # type: ignore

        # Sample noise and latent codes
        noise = torch.randn(batch_size, self.generator.latent_dim, device=device)
        fake_codes = self.encoder(real_data)
        real_codes = torch.randn(batch_size, self.generator.latent_dim, device=device)
        fake_data = self.generator(noise)
        fake_data_recon = self.generator(fake_codes)

        # Encoder loss and optimization
        recon_loss = self.reconstruction_loss(
            real_data=real_data, fake_data_recon=fake_data_recon
        )
        e_loss = recon_loss + self.encoder_loss(fake_codes)
        opt_e.zero_grad()
        self.manual_backward(e_loss)
        opt_e.step()

        # Generator loss and optimization
        g_loss = recon_loss + self.generator_loss(
            fake_data=fake_data, fake_data_recon=fake_data_recon
        )
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Discriminator loss and optimization
        d_loss = self.discriminator_loss(real_data, fake_data, real_codes)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # Code discriminator loss and optimization
        c_loss = self.code_discriminator_loss(real_codes, fake_codes)
        opt_c.zero_grad()
        self.manual_backward(c_loss)
        opt_c.step()

        # Log losses
        self.log_dict(
            {"d_loss": d_loss, "g_loss": g_loss, "c_loss": c_loss, "e_loss": e_loss}
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
