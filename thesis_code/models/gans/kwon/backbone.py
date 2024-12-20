import torch
import torch.nn as nn


class CodeDiscriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super(CodeDiscriminator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, code):
        return self.model(code)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_norm: bool = True,
    ):
        super(EncoderBlock, self).__init__()
        self.use_norm = use_norm

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.BatchNorm3d(out_channels) if self.use_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            # Layer 1
            EncoderBlock(1, 64, kernel_size=4, stride=2, padding=1, use_norm=False),
            # Layer 2
            EncoderBlock(64, 128, kernel_size=4, stride=2, padding=1, use_norm=True),
            # Layer 3
            EncoderBlock(128, 256, kernel_size=4, stride=2, padding=1, use_norm=True),
            # Layer 4
            EncoderBlock(256, 512, kernel_size=4, stride=2, padding=1, use_norm=True),
            # Layer 5
            nn.Conv3d(512, self.latent_dim, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Layer 1
            EncoderBlock(1, 64, kernel_size=4, stride=2, padding=1, use_norm=False),
            # Layer 2
            EncoderBlock(64, 128, kernel_size=4, stride=2, padding=1, use_norm=True),
            # Layer 3
            EncoderBlock(128, 256, kernel_size=4, stride=2, padding=1, use_norm=True),
            # Layer 4
            EncoderBlock(256, 512, kernel_size=4, stride=2, padding=1, use_norm=True),
            # Layer 5
            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.model(x)


class EncoderInBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=4, stride=1, padding=0
    ):
        super(EncoderInBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class EncoderMidBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1
    ):
        super(EncoderMidBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class EncoderOutBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1
    ):
        super(EncoderOutBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.Tanh(),
        )


class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # In layer
            EncoderInBlock(self.latent_dim, 512),
            # Mid layer 1
            EncoderMidBlock(512, 256),
            # Mid layer 2
            EncoderMidBlock(256, 128),
            # Mid layer 3
            EncoderMidBlock(128, 64),
            # Out layer
            EncoderOutBlock(64, 1),
        )

    def forward(self, x):
        return self.model(x)
