from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .layers import SNConv3d, SNLinear


class Code_Discriminator(nn.Module):
    def __init__(self, code_size, num_units=256):
        super(Code_Discriminator, self).__init__()

        self.l1 = nn.Sequential(
            SNLinear(code_size, num_units), nn.LeakyReLU(0.2, inplace=True)
        )
        self.l2 = nn.Sequential(
            SNLinear(num_units, num_units), nn.LeakyReLU(0.2, inplace=True)
        )
        self.l3 = SNLinear(num_units, 1)

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
    def __init__(self, num_class=0, channel=256):
        super(Sub_Discriminator, self).__init__()
        self.channel = channel
        self.num_class = num_class

        self.conv1 = SNConv3d(
            1, channel // 8, kernel_size=4, stride=2, padding=1
        )  # in:[64,64,64], out:[32,32,32]
        self.conv2 = SNConv3d(
            channel // 8, channel // 4, kernel_size=4, stride=2, padding=1
        )  # out:[16,16,16]
        self.conv3 = SNConv3d(
            channel // 4, channel // 2, kernel_size=4, stride=2, padding=1
        )  # out:[8,8,8]
        self.conv4 = SNConv3d(
            channel // 2, channel, kernel_size=4, stride=2, padding=1
        )  # out:[4,4,4]
        self.conv5 = SNConv3d(
            channel, 1 + num_class, kernel_size=4, stride=1, padding=0
        )  # out:[1,1,1,1]

    def forward(self, h):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        if self.num_class == 0:
            h = self.conv5(h).view((-1, 1))
            return h
        else:
            h = self.conv5(h).view((-1, 1 + self.num_class))
            return h[:, :1], h[:, 1:]


class Discriminator(nn.Module):
    def __init__(self, num_class=0, channel=512):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.num_class = num_class

        # D^H
        self.conv1 = SNConv3d(
            1, channel // 32, kernel_size=4, stride=2, padding=1
        )  # in:[32,256,256], out:[16,128,128]
        self.conv2 = SNConv3d(
            channel // 32, channel // 16, kernel_size=4, stride=2, padding=1
        )  # out:[8,64,64,64]
        self.conv3 = SNConv3d(
            channel // 16, channel // 8, kernel_size=4, stride=2, padding=1
        )  # out:[4,32,32,32]
        self.conv4 = SNConv3d(
            channel // 8,
            channel // 4,
            kernel_size=(2, 4, 4),
            stride=(2, 2, 2),
            padding=(0, 1, 1),
        )  # out:[2,16,16,16]
        self.conv5 = SNConv3d(
            channel // 4,
            channel // 2,
            kernel_size=(2, 4, 4),
            stride=(2, 2, 2),
            padding=(0, 1, 1),
        )  # out:[1,8,8,8]
        self.conv6 = SNConv3d(
            channel // 2,
            channel,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )  # out:[1,4,4,4]
        self.conv7 = SNConv3d(
            channel, channel // 4, kernel_size=(1, 4, 4), stride=1, padding=0
        )  # out:[1,1,1,1]
        self.fc1 = SNLinear(channel // 4 + 1, channel // 8)
        self.fc2 = SNLinear(channel // 8, 1)
        if num_class > 0:
            self.fc2_class = SNLinear(channel // 8, num_class)

        # D^L
        self.sub_D = Sub_Discriminator(num_class)

    def forward(self, h, h_small, crop_idx):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2)
        print("h0.shape:", h.shape)
        h = h.squeeze()
        print("h1.shape:", h.shape)
        print("h2.shape:", (crop_idx / 224.0 * torch.ones((h.size(0), 1))).shape)
        h = torch.cat(
            [h, (crop_idx / 224.0 * torch.ones((h.size(0), 1)))], 1
        )  # 256*7/8
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h)
        if self.num_class > 0:
            h_class_logit = self.fc2_class(h)

            h_small_logit, h_small_class_logit = self.sub_D(h_small)
            return (h_logit + h_small_logit) / 2.0, (
                h_class_logit + h_small_class_logit
            ) / 2.0
        else:
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
    def __init__(self, mode="train", latent_dim=1024, channel=32, num_class=0):
        super(Generator, self).__init__()
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.num_class = num_class

        # G^A and G^H
        self.fc1 = nn.Linear(latent_dim + num_class, 4 * 4 * 4 * _c * 16)

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

    def G_H(
        self, h_latent: torch.Tensor, crop_idx: Optional[int] = None
    ) -> torch.Tensor:
        if self.training:
            assert crop_idx is not None
            h = self.S_L(h_small=h_latent, crop_idx=crop_idx)
        else:
            h = h_latent

        # Generate from latent feature
        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h))  # (128, 128, 128)

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv7(h)

        h = torch.tanh(h)  # (256, 256, 256)

        return h

    def forward(
        self, h: torch.Tensor, crop_idx: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_latent = self.G_A(h)

        h_sub = self.G_H(h_latent, crop_idx=crop_idx)

        if self.training:
            h_small = self.G_L(h_latent)
            return h_sub, h_small
        return h_sub

    # def forward(self, h, crop_idx: Optional[int] = None):
    #     # Generate from random noise
    #     if crop_idx is not None or self.mode == "eval":
    #         # G_A
    #         h = self.fc1(h)

    #         h = h.view(-1, 512, 4, 4, 4)
    #         h = self.tp_conv1(h)
    #         h = self.relu(self.bn1(h))

    #         h = F.interpolate(h, scale_factor=2)
    #         h = self.tp_conv2(h)
    #         h = self.relu(self.bn2(h))

    #         h = F.interpolate(h, scale_factor=2)
    #         h = self.tp_conv3(h)
    #         h = self.relu(self.bn3(h))

    #         h = F.interpolate(h, scale_factor=2)
    #         h = self.tp_conv4(h)
    #         h = self.relu(self.bn4(h))

    #         h = F.interpolate(h, scale_factor=2)
    #         h = self.tp_conv5(h)
    #         h_latent = self.relu(self.bn5(h))  # (64, 64, 64), channel:128

    #         if self.mode == "train":
    #             # G_L
    #             h_small = self.sub_G(h_latent)

    #             # G_L
    #             h = h_latent[
    #                 :, :, crop_idx // 4 : crop_idx // 4 + 8, :, :
    #             ]  # Crop, out: (8, 64, 64)
    #         else:
    #             h = h_latent

    #     # Generate from latent feature
    #     h = F.interpolate(h, scale_factor=2)
    #     h = self.tp_conv6(h)
    #     h = self.relu(self.bn6(h))  # (128, 128, 128)

    #     h = F.interpolate(h, scale_factor=2)
    #     h = self.tp_conv7(h)

    #     h = torch.tanh(h)  # (256, 256, 256)

    #     if crop_idx is not None and self.mode == "train":
    #         return h, h_small
    #     return h

    def sample_z(self, num_samples: int) -> torch.Tensor:
        return torch.randn((num_samples, self.latent_dim), device=self.device)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        h_latent = self.G_A(z)

        h_generated = self.G_H(h_latent, crop_idx=None)
        return h_generated.detach()


def S_L(h_small: torch.Tensor, crop_idx: int) -> torch.Tensor:
    h = h_small[:, :, crop_idx // 4 : crop_idx // 4 + 8, :, :]  # Crop, out: (8, 64, 64)
    return h


def S_H(h: torch.Tensor, crop_idx: int) -> torch.Tensor:
    h_cropped = h[:, :, crop_idx : crop_idx + 256 // 8, :, :]
    return h_cropped
