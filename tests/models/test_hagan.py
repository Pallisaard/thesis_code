import pytest
import torch

from thesis_code.models.gans.hagan.backbone.Model_HA_GAN_256 import (
    Generator,
    Discriminator,
    Encoder,
    Sub_Encoder,
)

# Tests models downloaded from the internet. Really only interested in the output shapes being correct. You know, black box.


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def latent_dim():
    return 1024


@pytest.fixture
def generator(latent_dim, device):
    return Generator(latent_dim=latent_dim).to(device)


@pytest.fixture
def eval_encoder(generator):
    return generator.eval()


@pytest.fixture
def discriminator(device):
    return Discriminator().to(device)


@pytest.fixture
def encoder(device):
    return Encoder().to(device)


@pytest.fixture
def sub_encoder(latent_dim, device):
    return Sub_Encoder(latent_dim=latent_dim).to(device)


@pytest.fixture
def noise(latent_dim, device):
    return torch.randn((1, latent_dim), device=device)


def test_generator_output_shape(generator, latent_dim, device):
    noise = torch.randn((1, latent_dim), device=device)
    crop_idx = 0
    h_sub, h_small = generator(noise, crop_idx)
    assert h_sub.shape == (1, 1, 32, 256, 256)
    assert h_small.shape == (1, 1, 64, 64, 64)


def test_discriminator_output_shape(discriminator, device):
    real_images = torch.randn((2, 1, 32, 256, 256), device=device)
    real_images_small = torch.randn((2, 1, 64, 64, 64), device=device)
    crop_idx = 0
    output = discriminator(real_images, real_images_small, crop_idx)
    assert output.shape == (2, 1)


def test_encoder_output_shape(encoder, device):
    real_images_crop = torch.randn((2, 1, 256, 256, 256), device=device)
    output = encoder(real_images_crop)
    assert output.shape == (2, 64, 64, 64, 64)


def test_sub_encoder_output_shape(sub_encoder, device):
    real_images_small = torch.randn((2, 64, 64, 64, 64), device=device)
    output = sub_encoder(real_images_small)
    assert output.shape == (2, 1024)


def test_encoder_pipeline_output_shape(encoder, sub_encoder, device):
    real_images_crop = torch.randn((2, 1, 256, 256, 256), device=device)
    output = encoder(real_images_crop)
    output = sub_encoder(output)
    assert output.shape == (2, 1024)
