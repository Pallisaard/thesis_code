import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, sigma, z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, sigma
    
    def loss(self, recon_x, mu, sigma, x):
        # Reconstruction loss (e.g., MSE or BCE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        
        return recon_loss + kl_loss
    
if __name__ == '__main__':
    toy_X = torch.rand([1,1,32,32,32])
    model = VAE()
    recon_x, mu, sigma = model(toy_X)
    print(recon_x.shape)