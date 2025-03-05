import argparse
from pathlib import Path

import torch

from thesis_code.dataloading.utils import save_mri
from thesis_code.models.gans.hagan.hagan import LitHAGAN
from thesis_code.models.gans.hagan.backbone.Model_HA_GAN_256 import (
    Generator,
    Discriminator,
    Encoder,
    Sub_Encoder,
)
from thesis_code.models.gans.hagan.dp_safe_backbone.Model_HA_GAN_256 import (
    Generator as safe_Generator,
    Discriminator as safe_Discriminator,
    Encoder as safe_Encoder,
    Sub_Encoder as safe_Sub_Encoder,
)
from thesis_code.training.fine_tuning.utils import load_checkpoint_components


def load_custom_hagan_components(
    checkpoint_path: str,
    latent_dim: int = 1024,
    use_dp_safe: bool = False,
    map_location: str = "auto",
) -> tuple[
    Generator | safe_Generator,
    Discriminator | safe_Discriminator,
    Encoder | safe_Encoder,
    Sub_Encoder | safe_Sub_Encoder,
]:
    """Load individual components of HAGAN from a custom checkpoint.

    Args:
        checkpoint_path: Path to checkpoint containing individual components
        latent_dim: Dimension of latent space
        use_dp_safe: Whether to use DP safe version
        map_location: Device to map the model to

    Returns:
        Tuple containing:
        - generator: Generator network
        - encoder: Encoder network
        - discriminator: Discriminator network
        - code_discriminator: Code discriminator network
    """
    g_state_dict, d_state_dict, e_state_dict, sub_e_state_dict = load_checkpoint_components(
        checkpoint_path, map_location=map_location
    )

    if use_dp_safe:
        print("Using DP safe model")
        G = safe_Generator(latent_dim=latent_dim)
        D = safe_Discriminator()
        E = safe_Encoder()
        Sub_E = safe_Sub_Encoder(latent_dim=latent_dim)
    else:
        print("Using non-DP safe model")
        G = Generator(latent_dim=latent_dim)
        D = Discriminator()
        E = Encoder()
        Sub_E = Sub_Encoder(latent_dim=latent_dim)

    G.load_state_dict(g_state_dict)
    D.load_state_dict(d_state_dict)
    E.load_state_dict(e_state_dict)
    Sub_E.load_state_dict(sub_e_state_dict)

    return G, D, E, Sub_E


def load_hagan_from_components(
    generator: Generator | safe_Generator,
    discriminator: Discriminator | safe_Discriminator,
    encoder: Encoder | safe_Encoder,
    code_discriminator: Sub_Encoder | safe_Sub_Encoder,
    latent_dim: int = 1024,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    use_dp_safe: bool = False,
) -> LitHAGAN:
    """Create HAGAN model from individual components.

    Args:
        generator: Generator network
        encoder: Encoder network
        discriminator: Discriminator network
        code_discriminator: Code discriminator network
        latent_dim: Dimension of latent space
        lambda_1: Weight for first loss term
        lambda_2: Weight for second loss term
        use_dp_safe: Whether to use DP safe version

    Returns:
        Initialized HAGAN model with provided components
    """
    model = LitHAGAN(
        latent_dim=latent_dim,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        use_dp_safe=use_dp_safe,
    )

    model.G = generator
    model.E = encoder
    model.D = discriminator
    model.Sub_E = code_discriminator

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MRI samples from a trained HA-GAN model")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save the generated samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples to generate (default: 2)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=1024,
        help="Dimension of the latent space (default: 1024)",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=5.0,
        help="Lambda 1 parameter (default: 5.0)",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=5.0,
        help="Lambda 2 parameter (default: 5.0)",
    )
    parser.add_argument(
        "--use-dp-safe",
        action="store_true",
        help="Use DP safe.",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default="0000",
        help="Name of the sample (default: 0000)",
    )
    parser.add_argument(
        "--use-custom-checkpoint",
        action="store_true",
        help="Load custom checkpoint with individual HAGAN components",
    )
    parser.add_argument(
        "--custom-checkpoint-map-location",
        type=str,
        help="Map location for custom checkpoint",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.checkpoint_path}")
    if args.use_custom_checkpoint:
        components = load_custom_hagan_components(
            args.checkpoint_path,
            map_location=args.custom_checkpoint_map_location,
            use_dp_safe=args.use_dp_safe,
        )
        model = load_hagan_from_components(
            *components,
            latent_dim=args.latent_dim,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            use_dp_safe=args.use_dp_safe,
        )
    else:
        model = LitHAGAN.load_from_checkpoint(
            args.checkpoint_path,
            map_location="cpu",
            latent_dim=args.latent_dim,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            use_dp_safe=args.use_dp_safe,
        )

    print("Model loaded")
    model.eval()

    print("Sampling")
    with torch.no_grad():
        samples = model.sample(args.num_samples)

    print(f"Samples shape: {samples.shape}")

    print(f"Saving samples to {args.save_path}")
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Saving samples")

    save_mri(samples, save_path / f"sample_{args.sample_name}.nii.gz")


if __name__ == "__main__":
    main()
