import argparse
from pathlib import Path

import nibabel as nib
import torch
import tqdm
from monai.metrics.regression import MultiScaleSSIMMetric

from thesis_code.models import LitKwonGan, LitWGANGP, LitAlphaGAN, LitVAE3D, LitHAGAN
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


def get_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    latent_dim: int = 1024,
    lambdas: float = 1.0,
    use_dp_safe: bool = False,
    use_custom_checkpoint: bool = False,
    map_location: str = "auto",
) -> torch.nn.Module:
    """Load a model from a checkpoint file."""
    if model_name == "cicek_3d_vae_256":
        return LitVAE3D.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            in_shape=(1, 256, 256, 256),
            encoder_out_channels_per_block=[8, 16, 32, 64],
            decoder_out_channels_per_block=[64, 64, 16, 8, 1],
            latent_dim=latent_dim,
        )
    elif model_name == "cicek_3d_vae_64":
        return LitVAE3D.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            in_shape=(1, 64, 64, 64),
            encoder_out_channels_per_block=[16, 32, 64],
            decoder_out_channels_per_block=[64, 32, 16, 1],
            latent_dim=latent_dim,
        )
    elif model_name == "alpha_gan":
        return LitAlphaGAN.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
        )
    elif model_name == "wgan_gp":
        return LitWGANGP.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
        )
    elif model_name == "kwon_gan":
        return LitKwonGan.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            lambda_recon=lambdas,
            lambda_gp=lambdas,
        )
    elif model_name == "hagan":
        if use_custom_checkpoint:
            components = load_custom_hagan_components(
                checkpoint_path,
                map_location=map_location,
                use_dp_safe=use_dp_safe,
            )
            return load_hagan_from_components(
                *components,
                latent_dim=latent_dim,
                lambda_1=lambdas,
                lambda_2=lambdas,
                use_dp_safe=use_dp_safe,
            )
        return LitHAGAN.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            lambda_1=lambdas,
            lambda_2=lambdas,
            use_dp_safe=use_dp_safe,
        )
    else:
        raise ValueError(f"Model name {model_name} not recognized")


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
    """Load individual components of HAGAN from a custom checkpoint."""
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
    """Create HAGAN model from individual components."""
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
    parser = argparse.ArgumentParser(description="Compute average MS-SSIM between pairs of generated MRIs")

    # Input options - either directory of samples or model checkpoint
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing generated MRI samples (sample_*.nii.gz files)",
    )
    input_group.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to model checkpoint for generating samples on the fly",
    )

    # Required arguments
    parser.add_argument("--device", required=True, type=str, help="Device to use")
    parser.add_argument("--output-file", required=True, type=str, help="Path to save the MS-SSIM scores")
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[64, 256],
        required=True,
        help="Resolution of the MRI volumes (64 or 256)",
    )

    # Optional arguments for model-based generation
    parser.add_argument(
        "--model-name",
        choices=[
            "cicek_3d_vae_64",
            "cicek_3d_vae_256",
            "kwon_gan",
            "wgan_gp",
            "alpha_gan",
            "hagan",
        ],
        help="Name of the model to load",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate when using checkpoint",
    )
    parser.add_argument(
        "--use-small-model",
        action="store_true",
        help="Use sub generator of HAGAN",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        help="Value for lambda_1 and lambda_2",
    )
    parser.add_argument(
        "--use-dp-safe",
        action="store_true",
        help="Use DP safe version of model",
    )
    parser.add_argument(
        "--use-custom-checkpoint",
        action="store_true",
        help="Load custom checkpoint with individual HAGAN components",
    )
    parser.add_argument(
        "--custom-checkpoint-map-location",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Map location for custom checkpoint",
    )

    return parser.parse_args()


def generate_and_compute_msssim(model, ms_ssim, n_pairs: int, device: str, use_small_model: bool) -> list:
    """Generate samples in pairs and compute MS-SSIM scores on the fly."""
    ms_ssim_scores = []

    with torch.no_grad():
        pbar = tqdm.tqdm(range(n_pairs), desc="Generating pairs and computing MS-SSIM")
        for _ in pbar:
            # Generate a batch of 2 samples
            if use_small_model:
                batch = model.sample_small(2)
            else:
                batch = model.sample(2)

            # Compute MS-SSIM on the pair
            score = ms_ssim(batch[0:1].to(device), batch[1:2].to(device))
            ms_ssim_scores.append(score.item())  # type: ignore

    return ms_ssim_scores


def main():
    args = parse_args()
    print("Computing MS-SSIM diversity scores")
    print(vars(args))

    # Setup metric with kernel size based on resolution
    kernel_size = 3 if args.resolution == 64 else 11
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, kernel_size=kernel_size)

    # Get samples either from files or generate them
    if args.input_dir:
        # Get all sample files
        sample_files = sorted(Path(args.input_dir).glob("sample_*.nii.gz"))
        print(f"Found {len(sample_files)} samples")

        # Get all unique pairs
        pairs = [(sample_files[i], sample_files[i + 1]) for i in range(0, len(sample_files) - 1, 2)]

        # Process pairs with progress bar
        pbar = tqdm.tqdm(pairs, desc="Computing pairwise MS-SSIM")
        ms_ssim_scores = []

        for file1, file2 in pbar:
            # Load and normalize MRIs
            mri1 = nib.load(file1).get_fdata()  # type: ignore
            mri2 = nib.load(file2).get_fdata()  # type: ignore

            # Convert to tensors
            mri1 = torch.from_numpy(mri1).unsqueeze(0).unsqueeze(0).to(args.device)
            mri2 = torch.from_numpy(mri2).unsqueeze(0).unsqueeze(0).to(args.device)

            # Compute MS-SSIM
            score = ms_ssim(mri1, mri2)
            ms_ssim_scores.append(score.item())  # type: ignore
    else:
        # Generate samples on the fly
        if not args.model_name:
            raise ValueError("--model-name is required when using --checkpoint-path")

        print("Loading model for sample generation")
        model = (
            get_model_from_checkpoint(
                model_name=args.model_name,
                checkpoint_path=args.checkpoint_path,
                latent_dim=1024,
                lambdas=args.lambdas,
                use_dp_safe=args.use_dp_safe,
                use_custom_checkpoint=args.use_custom_checkpoint,
                map_location=args.custom_checkpoint_map_location,
            )
            .eval()
            .to(args.device)
        )

        n_pairs = args.n_samples // 2
        print(f"Will generate {n_pairs} pairs of samples")
        ms_ssim_scores = generate_and_compute_msssim(
            model=model,
            ms_ssim=ms_ssim,
            n_pairs=n_pairs,
            device=args.device,
            use_small_model=args.use_small_model,
        )

    # Calculate and print average
    average_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
    print(f"Average MS-SSIM: {average_ms_ssim:.4f}")

    # Save scores
    torch.save(torch.tensor(ms_ssim_scores), args.output_file)


if __name__ == "__main__":
    main()
