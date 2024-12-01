from functools import reduce
import pytest
import torch

from thesis_code.models.vaes.cicek_3d_vae import (
    ConvUnit,
    ConvEncoderBlock,
    VAE3DEncoder,
    ConvDecoderBlock,
    VAE3DDecoder,
    VAE3D,
    ResNetBlock3D,
)


@pytest.fixture
def base_shape():
    # Batch size of 2, 1 channel, 64x64x64 volume
    return (2, 1, 64, 64, 64)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_conv_unit(base_shape, device):
    in_channels = 1
    out_channels = 16

    model = ConvUnit(in_channels, out_channels).to(device)
    x = torch.randn(*base_shape).to(device)

    output = model(x)
    expected_shape = (2, out_channels, 64, 64, 64)

    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_conv_encoder_block(base_shape, device):
    channels = [1, 16, 32]  # in -> hidden -> out
    pool_size = 2

    model = ConvEncoderBlock(channels, pool_size=pool_size).to(device)
    x = torch.randn(*base_shape).to(device)

    output = model(x)
    expected_shape = (2, channels[-1], 32, 32, 32)  # Size halved due to pooling

    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_vae_3d_encoder(base_shape, device):
    out_channels_per_block = [16, 32, 64, 128]  # Example channel progression
    pool_size = 2

    model = VAE3DEncoder(
        out_channels_per_block=out_channels_per_block,
        pool_size=pool_size,
        in_shape=base_shape[1:],  # Remove batch dimension
    ).to(device)

    x = torch.randn(*base_shape).to(device)
    output = model(x)

    # Calculate expected flattened size
    n_pools = len(out_channels_per_block)
    final_spatial_dim = 64 // (pool_size**n_pools)
    expected_flatten_size = (2, out_channels_per_block[-1] * final_spatial_dim**3)

    assert output.shape == expected_flatten_size
    assert not torch.isnan(output).any()


def test_conv_decoder_block(device):
    channels = [128, 64, 32]  # in -> hidden -> out
    kernel_size = 2
    stride = 2

    # Input shape after previous operations
    input_shape = (2, channels[0], 8, 8, 8)

    model = ConvDecoderBlock(channels, kernel_size=kernel_size, stride=stride).to(
        device
    )

    x = torch.randn(*input_shape).to(device)
    output = model(x)

    expected_shape = (
        2,
        channels[-1],
        16,
        16,
        16,
    )  # Size doubled due to transposed conv

    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_vae_3d_decoder(device):
    decoder_channels = [128, 64, 32, 16, 1]  # Progression back to original channels
    input_shape = (2, decoder_channels[0], 4, 4, 4)  # Example encoded shape

    model = VAE3DDecoder(
        out_channels_per_block=decoder_channels,
        in_shape=input_shape[1:],  # Remove batch dimension
        kernel_size=2,
        stride=2,
    ).to(device)

    latent_size = reduce(lambda x, y: x * y, input_shape[1:], 1)
    z = torch.randn(2, latent_size).to(device)
    output = model(z)

    expected_shape = (
        2,
        decoder_channels[-1],
        64,
        64,
        64,
    )  # Back to original spatial dims

    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_full_vae_3d(base_shape, device):
    encoder_channels = [16, 32, 64, 128]
    decoder_channels = [128, 64, 32, 16, 1]
    latent_dim = 256

    model = VAE3D(
        in_shape=base_shape[1:],  # Remove batch dimension
        encoder_out_channels_per_block=encoder_channels,
        decoder_out_channels_per_block=decoder_channels,
        latent_dim=latent_dim,
        pool_size=2,
        kernel_size=2,
        stride=2,
    ).to(device)

    x = torch.clip(torch.randn(*base_shape), 0, 1).to(device)
    recon_x, mu, log_var = model(x)

    # Test shapes
    assert recon_x.shape == base_shape
    assert mu.shape == (base_shape[0], latent_dim)
    assert log_var.shape == (base_shape[0], latent_dim)

    # Test values
    assert not torch.isnan(recon_x).any()
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_var).any()

    # Test loss calculation
    loss, recon_loss, kld_loss = model.calculate_loss(x, recon_x, mu, log_var, 1)
    assert not torch.isnan(loss).item()
    assert not torch.isnan(recon_loss).item()
    assert not torch.isnan(kld_loss).item()


# @pytest.fixture
# def resnet_block_input_shape():
#     # Batch size of 2, 16 channels, 32x32x32 volume
#     return (2, 16, 32, 32, 32)


# def test_resnet_block_3d(resnet_block_input_shape, device):
#     in_channels = 16
#     out_channels = 32
#     stride = 1

#     model = ResNetBlock3D(in_channels, out_channels, stride=stride).to(device)
#     x = torch.randn(*resnet_block_input_shape).to(device)

#     output = model(x)
#     expected_shape = (2, out_channels, 32, 32, 32)

#     assert isinstance(output, torch.Tensor)
#     assert output.shape == expected_shape
#     assert not torch.isnan(output).any()
