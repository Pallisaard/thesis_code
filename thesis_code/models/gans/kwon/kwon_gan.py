import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
)
from .backbone import CodeDiscriminator, Encoder, Generator, Discriminator


class LitKwonGan(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lambda_gp: float = 10.0,
        n_critic_steps: int = 5,
        lambda_recon: float = 1.0,
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

        self.ssim = StructuralSimilarityIndexMeasure(
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
        )

        self.automatic_optimization = False  # Manual optimization

    def verify_models(self, x):
        z = self.sample_z(x.size(0))
        assert self.generator(z).shape == x.shape
        assert self.critic(x).shape == (x.shape[0], 1)
        assert self.code_critic(z).shape == (z.shape[0], 1)
        assert self.encoder(x).shape == z.shape

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def sample_mri(self) -> torch.Tensor:
        return self(self.sample_z(1))[0]

    def sample_n_mri(self, n) -> torch.Tensor:
        return self(self.sample_z(n))

    def forward(self, z):
        return self.generator(z)

    def reconstruction_loss(
        self, real_data: torch.Tensor, recon_data: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(real_data, recon_data)

    def generator_loss(
        self,
        real_critic_score: torch.Tensor,
        fake_critic_score: torch.Tensor,
        real_data: torch.Tensor,
        recon_data: torch.Tensor,
    ) -> torch.Tensor:
        return (
            -torch.mean(fake_critic_score)
            - torch.mean(real_critic_score)
            + self.lambda_recon * self.reconstruction_loss(real_data, recon_data)
        )

    def critic_loss(
        self,
        real_critic_score: torch.Tensor,
        fake_critic_score: torch.Tensor,
        recon_critic_score: torch.Tensor,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> torch.Tensor:
        gp_loss = self._gradient_policy(self.critic, real_data, fake_data)
        return (
            torch.mean(fake_critic_score)
            + torch.mean(recon_critic_score)
            - 2.0 * torch.mean(real_critic_score)
            + self.lambda_gp * gp_loss
        )

    def code_critic_loss(
        self,
        real_code_critic_score: torch.Tensor,
        fake_code_critic_score: torch.Tensor,
        random_codes: torch.Tensor,
        fake_codes: torch.Tensor,
    ) -> torch.Tensor:
        gp_loss = self._gradient_policy(self.code_critic, random_codes, fake_codes)

        return (
            torch.mean(fake_code_critic_score)
            - torch.mean(real_code_critic_score)
            + self.lambda_gp * gp_loss
        )

    def encoder_loss(
        self,
        fake_code_critic_score: torch.Tensor,
        real_data: torch.Tensor,
        recon_data: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean(fake_code_critic_score)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # Optimizers
        opt_g, opt_d, opt_c, opt_e = self.optimizers()  # type: ignore

        real_data = batch
        batch_size = real_data.size(0)
        latent_dim = self.generator.latent_dim

        # Sample noise and latent codes
        random_codes = torch.randn(batch_size, latent_dim, device=self.device)
        latent_codes = self.encoder(real_data)
        # All generated data
        fake_data = self.generator(random_codes)
        recon_data = self.generator(latent_codes)
        # All scores from the critic, should be scalars
        real_critic_score = self.critic(real_data)
        fake_critic_score = self.critic(fake_data)
        recon_critic_score = self.critic(recon_data)
        # All scores from the code critic, should be scalars
        real_code_critic_score = self.code_critic(random_codes)
        fake_code_critic_score = self.code_critic(latent_codes)

        # Encoder loss and optimization
        e_loss = self.encoder_loss(
            fake_code_critic_score=fake_code_critic_score,
            real_data=real_data,
            recon_data=recon_data,
        )
        opt_e.zero_grad()
        # Also performs backwards for reconstruction loss on generator, so no
        # need to call it again
        self.manual_backward(e_loss, retain_graph=True)
        opt_e.step()

        # Generator loss and optimization
        g_loss = self.generator_loss(
            real_critic_score=real_critic_score,
            fake_critic_score=fake_critic_score,
            real_data=real_data,
            recon_data=recon_data,
        )
        opt_g.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        opt_g.step()

        # critic loss and optimization
        d_loss = self.critic_loss(
            real_critic_score=real_critic_score,
            fake_critic_score=fake_critic_score,
            recon_critic_score=recon_critic_score,
            real_data=real_data,
            fake_data=fake_data,
        )
        opt_d.zero_grad()
        self.manual_backward(d_loss, retain_graph=True)
        opt_d.step()

        # Code critic loss and optimization
        c_loss = self.code_critic_loss(
            real_code_critic_score=real_code_critic_score,
            fake_code_critic_score=fake_code_critic_score,
            random_codes=random_codes,
            fake_codes=latent_codes,
        )
        opt_c.zero_grad()
        self.manual_backward(c_loss, retain_graph=True)
        opt_c.step()

        # Log losses
        self.log_dict(
            {"d_loss": d_loss, "g_loss": g_loss, "c_loss": c_loss, "e_loss": e_loss}
        )

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        real_data = batch
        batch_size = real_data.size(0)
        latent_dim = self.generator.latent_dim

        # Sample noise and latent codes
        random_codes = torch.randn(batch_size, latent_dim, device=self.device)
        latent_codes = self.encoder(real_data)
        # All generated data
        fake_data = self.generator(random_codes)
        recon_data = self.generator(latent_codes)
        # All scores from the critic, should be scalars
        real_critic_score = self.critic(real_data)
        fake_critic_score = self.critic(fake_data)
        recon_critic_score = self.critic(recon_data)
        # All scores from the code critic, should be scalars
        real_code_critic_score = self.code_critic(random_codes)
        fake_code_critic_score = self.code_critic(latent_codes)

        # Enable gradient computation for gradient penalty calculation
        with torch.enable_grad():
            # Encoder loss and optimization
            e_loss = self.encoder_loss(
                fake_code_critic_score=fake_code_critic_score,
                real_data=real_data,
                recon_data=recon_data,
            )

            # Generator loss and optimization
            g_loss = self.generator_loss(
                real_critic_score=real_critic_score,
                fake_critic_score=fake_critic_score,
                real_data=real_data,
                recon_data=recon_data,
            )

            # Critic loss and optimization
            d_loss = self.critic_loss(
                real_critic_score=real_critic_score,
                fake_critic_score=fake_critic_score,
                recon_critic_score=recon_critic_score,
                real_data=real_data,
                fake_data=fake_data,
            )

            # Code critic loss and optimization
            c_loss = self.code_critic_loss(
                real_code_critic_score=real_code_critic_score,
                fake_code_critic_score=fake_code_critic_score,
                random_codes=random_codes,
                fake_codes=latent_codes,
            )

        self.ssim(recon_data, real_data)

        # Log losses
        self.log_dict(
            {
                "val_d_loss": d_loss,
                "val_g_loss": g_loss,
                "val_c_loss": c_loss,
                "val_e_loss": e_loss,
                "val_total_loss": d_loss + g_loss + c_loss + e_loss,
                "val_ssim": self.ssim,
            }
        )

    def configure_optimizers(self):
        # Separate optimizers for generator, critic, and code critic
        opt_g = torch.optim.Adam(self.generator.parameters())
        opt_d = torch.optim.Adam(self.critic.parameters())
        opt_c = torch.optim.Adam(self.code_critic.parameters())
        opt_e = torch.optim.Adam(self.encoder.parameters())
        return [opt_g, opt_d, opt_c, opt_e]

    def _gradient_policy(
        self,
        model: nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> torch.Tensor:
        print("In gradient policy fn")
        print("model name:", model.__class__.__name__)
        print("real_data.size():", real_data.size())
        print("fake_data.size():", fake_data.size())
        interpolates_grad = self._interpolate_data_with_gradient(real_data, fake_data)

        batch_size = fake_data.size(0)
        # Computes the gradient of the critic with respect to the fake data

        gradient = self._compute_gradient(model, interpolates_grad)

        # Flatten so the norm computation is easier
        gradient = gradient.view(batch_size, -1)
        # Norm with added epsilon to avoid division by zero
        gradient_norms = torch.sqrt(torch.sum(gradient**2, dim=1) + 1e-12)
        # Compute the gradient penalty

        return torch.mean((gradient_norms - 1) ** 2)

    def _compute_gradient(self, model: nn.Module, interpolates_grad: torch.Tensor):
        interpolates_grad.requires_grad_(True)
        gradient = torch.autograd.grad(
            outputs=model(interpolates_grad).sum(),
            inputs=interpolates_grad,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        return gradient

    def _interpolate_data_with_gradient(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        alpha_dim = (real_data.size(0), *([1] * (real_data.dim() - 1)))
        alpha = torch.rand(size=alpha_dim, device=self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        return interpolates
