import os
import warnings

import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.optim import SGD

from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.lightning import DPLightningDataModule





input_shape = (1, 32, 32, 32)
z_dim = 128

class Sampling(nn.Module):
    def forward(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        return mu + torch.exp(0.5 * sigma) * epsilon

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(4, 8),  
            nn.ELU()
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.ELU()
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(16, 32),
            nn.ELU()
        )
        self.enc_conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.ELU()
        )
        self.flatten = nn.Flatten()
        self.enc_fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7 * 7, 343),
            nn.GroupNorm(1, 343),  
            nn.ELU()
        )
        self.mu = nn.Linear(343, z_dim)
        self.sigma = nn.Linear(343, z_dim)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)
        
        x = self.flatten(x)
        x = self.enc_fc1(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = self.sampling(mu, sigma)
        return mu, sigma, z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_fc1 = nn.Sequential(
            nn.Linear(z_dim, 343),
            nn.GroupNorm(1, 343),  
            nn.ELU()
        )
        self.dec_unflatten = nn.Unflatten(1, (1, 7, 7, 7))

        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 64),
            nn.ELU()
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(16, 32),
            nn.ELU()
        )
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.ELU()
        )
        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=0),
            nn.GroupNorm(4, 8),
            nn.ELU()
        )
        self.dec_conv5 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 1)  
        )

    def forward(self, z):
        x = self.dec_fc1(z)
        x = self.dec_unflatten(x)
        x = self.dec_conv1(x)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)
        x = self.dec_conv4(x)
        x = self.dec_conv5(x)
        return x


class VAE_Lightning_DP(L.LightningModule):
    def __init__(self, enable_dp = True, target_epsilon=1.0, delta=1e-5, max_grad_norm=1.0, sample_rate=0.01):
        super(VAE_Lightning_DP, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.learning_rate = 1e-3

        # Parameters for the PrivacyEngine
        self.enable_dp = enable_dp
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.sample_rate = sample_rate
        self.noise_multiplier = 1.0
        self.privacy_engine = PrivacyEngine() if self.enable_dp else None
        

    def forward(self, x):
        mu, sigma, z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, sigma

    def vae_loss(self, recon_x, x, mu, sigma):
        # Reconstruction loss (e.g., MSE or BCE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        
        return recon_loss + kl_loss

    def training_step(self, batch, batch_idx):
        x = batch  # Assuming the dataset returns (data, target) pairs
        recon_x, mu, sigma = self(x)
        print(recon_x[0])
        loss = self.vae_loss(recon_x, x, mu, sigma)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self,batch,batch_idx):
        x = batch  # Assuming the dataset returns (data, target) pairs
        recon_x, mu, sigma = self(x)
        loss = self.vae_loss(recon_x, x, mu, sigma)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.learning_rate)
        if self.enable_dp:
            self.trainer.fit_loop.setup_data()
            data_loader = (
                # soon there will be a fancy way to access train dataloader,
                # see https://github.com/PyTorchLightning/pytorch-lightning/issues/10430
                
                
                self.trainer.train_dataloader
                #self.trainer._data_connector._train_dataloader_source.dataloader()
            )
            if hasattr(self, "dp"):
                self.dp["model"].remove_hooks()
            dp_model, optimizer, dataloader = self.privacy_engine.make_private(
                module=self,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=isinstance(data_loader, DPDataLoader),
            )
            self.dp = {"model": dp_model}
        
        return optimizer

    def on_train_epoch_end(self):
        # Logging privacy statistics
        if self.enable_dp:
            epsilon = self.privacy_engine.get_epsilon(self.delta)
            self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)

    def on_train_end(self):
        # Detach the privacy engine at the end of training
        #self.privacy_engine.detach()
        pass
if __name__ == '__main__':
    # Example of how to use the model
    input_data = torch.randn((1, 1, 32, 32, 32))  # Example input
    model = VAE_Lightning_DP(enable_dp=False)
    output, mu, sigma = model(input_data)

    print("Output shape:", output.shape)
    print("Mu shape:", mu.shape)
    print("Sigma shape:", sigma.shape)

