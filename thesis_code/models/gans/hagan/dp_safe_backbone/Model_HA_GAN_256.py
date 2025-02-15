from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from .layers import SnLinear, SnConv3d


class Code_Discriminator(nn.Module):
    def __init__(self, code_size, num_units=256):
        super(Code_Discriminator, self).__init__()

        self.l1 = nn.Sequential(
            SnLinear(code_size, num_units), nn.LeakyReLU(0.2, inplace=True)
        )
        self.l2 = nn.Sequential(
            SnLinear(num_units, num_units), nn.LeakyReLU(0.2, inplace=True)
        )
        self.l3 = SnLinear(num_units, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class Sub_Encoder(nn.Module):
    def __init__(self, channel=256, latent_dim=1024):
        super(Sub_Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(
            channel // 4, channel // 8, kernel_size=4, stride=2, padding=1
        )  # in:[64,64,64], out:[32,32,32]
        self.bn1 = nn.GroupNorm(8, channel // 8)
        self.conv2 = nn.Conv3d(
            channel // 8, channel // 4, kernel_size=4, stride=2, padding=1
        )  # out:[16,16,16]
        self.bn2 = nn.GroupNorm(8, channel // 4)
        self.conv3 = nn.Conv3d(
            channel // 4, channel // 2, kernel_size=4, stride=2, padding=1
        )  # out:[8,8,8]
        self.bn3 = nn.GroupNorm(8, channel // 2)
        self.conv4 = nn.Conv3d(
            channel // 2, channel, kernel_size=4, stride=2, padding=1
        )  # out:[4,4,4]
        self.bn4 = nn.GroupNorm(8, channel)
        self.conv5 = nn.Conv3d(
            channel, latent_dim, kernel_size=4, stride=1, padding=0
        )  # out:[1,1,1,1]

    def forward(self, h):
        h = self.conv1(h)
        h = self.relu(self.bn1(h))
        h = self.conv2(h)
        h = self.relu(self.bn2(h))
        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        h = self.conv4(h)
        h = self.relu(self.bn4(h))
        h = self.conv5(h).squeeze()
        return h


class Encoder(nn.Module):
    def __init__(self, channel=64):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(
            1, channel // 2, kernel_size=4, stride=2, padding=1
        )  # in:[32,256,256], out:[16,128,128]
        self.bn1 = nn.GroupNorm(8, channel // 2)
        self.conv2 = nn.Conv3d(
            channel // 2, channel // 2, kernel_size=3, stride=1, padding=1
        )  # out:[16,128,128]
        self.bn2 = nn.GroupNorm(8, channel // 2)
        self.conv3 = nn.Conv3d(
            channel // 2, channel, kernel_size=4, stride=2, padding=1
        )  # out:[8,64,64]
        self.bn3 = nn.GroupNorm(8, channel)

    def forward(self, h):
        h = self.conv1(h)
        h = self.relu(self.bn1(h))

        h = self.conv2(h)
        h = self.relu(self.bn2(h))

        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        return h


# D^L
class Sub_Discriminator(nn.Module):
    def __init__(self, channel=256):
        super(Sub_Discriminator, self).__init__()
        self.channel = channel

        self.conv1 = SnConv3d(
            1, channel // 8, kernel_size=4, stride=2, padding=1
        )  # in:[64,64,64], out:[32,32,32]
        self.conv2 = SnConv3d(
            channel // 8, channel // 4, kernel_size=4, stride=2, padding=1
        )  # out:[16,16,16]
        self.conv3 = SnConv3d(
            channel // 4, channel // 2, kernel_size=4, stride=2, padding=1
        )  # out:[8,8,8]
        self.conv4 = SnConv3d(
            channel // 2, channel, kernel_size=4, stride=2, padding=1
        )  # out:[4,4,4]
        self.conv5 = SnConv3d(
            channel, 1, kernel_size=4, stride=1, padding=0
        )  # out:[1,1,1,1]

    def forward(self, h):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = self.conv5(h).view((-1, 1))
        return h


class Discriminator(nn.Module):
    def __init__(self, channel=512):
        super(Discriminator, self).__init__()
        self.channel = channel

        # D^H
        self.conv1 = SnConv3d(
            1, channel // 32, kernel_size=4, stride=2, padding=1
        )  # in:[32,256,256], out:[16,128,128]
        self.conv2 = SnConv3d(
            channel // 32, channel // 16, kernel_size=4, stride=2, padding=1
        )  # out:[8,64,64,64]
        self.conv3 = SnConv3d(
            channel // 16, channel // 8, kernel_size=4, stride=2, padding=1
        )  # out:[4,32,32,32]
        self.conv4 = SnConv3d(
            channel // 8,
            channel // 4,
            kernel_size=(2, 4, 4),
            stride=(2, 2, 2),
            padding=(0, 1, 1),
        )  # out:[2,16,16,16]
        self.conv5 = SnConv3d(
            channel // 4,
            channel // 2,
            kernel_size=(2, 4, 4),
            stride=(2, 2, 2),
            padding=(0, 1, 1),
        )  # out:[1,8,8,8]
        self.conv6 = SnConv3d(
            channel // 2,
            channel,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )  # out:[1,4,4,4]
        self.conv7 = SnConv3d(
            channel, channel // 4, kernel_size=(1, 4, 4), stride=1, padding=0
        )  # out:[1,1,1,1]
        self.fc1 = SnLinear(channel // 4 + 1, channel // 8)
        self.fc2 = SnLinear(channel // 8, 1)

        # D^L
        self.sub_D = Sub_Discriminator()

    def forward(self, h, h_small, crop_idx):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2)
        h = h.squeeze()
        if h.ndim == 1:
            h = h.unsqueeze(0)
        h = torch.cat(
            [h, (crop_idx / 224.0 * torch.ones((h.size(0), 1), device=h.device))], 1
        )  # 256*7/8
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h)

        h_small_logit = self.sub_D(h_small)
        return (h_logit + h_small_logit) / 2.0


class Sub_Generator(nn.Module):
    def __init__(self, channel: int = 16):
        super(Sub_Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.tp_conv1 = nn.Conv3d(
            _c * 4, _c * 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn1 = nn.GroupNorm(8, _c * 2)

        self.tp_conv2 = nn.Conv3d(
            _c * 2, _c, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.GroupNorm(8, _c)

        self.tp_conv3 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, h):
        h = self.tp_conv1(h)
        h = self.relu(self.bn1(h))

        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = self.tp_conv3(h)
        h = torch.tanh(h)
        return h


class Generator(nn.Module):
    def __init__(self, latent_dim=1024, channel=32):
        super(Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()

        # G^A and G^H
        self.fc1 = nn.Linear(latent_dim, 4 * 4 * 4 * _c * 16)

        self.tp_conv1 = nn.Conv3d(
            _c * 16, _c * 16, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn1 = nn.GroupNorm(8, _c * 16)

        self.tp_conv2 = nn.Conv3d(
            _c * 16, _c * 16, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.GroupNorm(8, _c * 16)

        self.tp_conv3 = nn.Conv3d(
            _c * 16, _c * 8, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn3 = nn.GroupNorm(8, _c * 8)

        self.tp_conv4 = nn.Conv3d(
            _c * 8, _c * 4, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn4 = nn.GroupNorm(8, _c * 4)

        self.tp_conv5 = nn.Conv3d(
            _c * 4, _c * 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn5 = nn.GroupNorm(8, _c * 2)

        self.tp_conv6 = nn.Conv3d(
            _c * 2, _c, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn6 = nn.GroupNorm(8, _c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # G^L
        self.sub_G = Sub_Generator(channel=_c // 2)

    def G_A(self, h: torch.Tensor) -> torch.Tensor:
        h = self.fc1(h)

        h = h.view(-1, 512, 4, 4, 4)
        h = self.tp_conv1(h)
        h = self.relu(self.bn1(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv5(h)
        h_latent = self.relu(self.bn5(h))  # (64, 64, 64), channel:128
        return h_latent

    def G_L(self, h_latent: torch.Tensor) -> torch.Tensor:
        h_small = self.sub_G(h_latent)
        return h_small

    def G_H(self, h: torch.Tensor) -> torch.Tensor:
        # Generate from latent feature
        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h))  # (128, 128, 128)

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv7(h)

        h = torch.tanh(h)  # (256, 256, 256)

        return h

    def forward(
        self, h: torch.Tensor, crop_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_small_image = self.G_A(h)
        h_small_image_cropped = S_L(h_small=h_small_image, crop_idx=crop_idx)
        h_sub = self.G_H(h_small_image_cropped)
        h_small = self.G_L(h_small_image)
        return h_sub, h_small

    def sample_z(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.latent_dim), device=self.device)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        h_latent = self.G_A(z)

        h_generated = self.G_H(h_latent)
        return h_generated.detach()

    def sample(self, num_samples: int) -> torch.Tensor:
        z = self.sample_z(num_samples)
        return self.generate(z)


def S_L(h_small: torch.Tensor, crop_idx: int) -> torch.Tensor:
    h = h_small[:, :, crop_idx // 4 : crop_idx // 4 + 8, :, :]  # Crop, out: (8, 64, 64)
    return h


def S_H(h: torch.Tensor, crop_idx: int) -> torch.Tensor:
    h_cropped = h[:, :, crop_idx : crop_idx + 256 // 8, :, :]
    return h_cropped
