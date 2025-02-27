import argparse
from itertools import batched
from pathlib import Path

import nibabel as nib
import torch
import numpy as np
import tqdm
import lightning as L

from thesis_code.metrics.utils import get_mri_vectorizer
from thesis_code.training.utils import numpy_to_nifti
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
from thesis_code.training.fine_tuning.utils import tree_key_map


def pars_args():
    parser = argparse.ArgumentParser(description="Generate n sampled MRIs")

    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--n-samples", required=True, type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "--from-authors",
        action="store_true",
        help="Generate samples from authors model",
    )
    parser.add_argument(
        "--use-small-model",
        action="store_true",
        help="Use sub generator of HAGAN",
    )
    parser.add_argument(
        "--checkpoint-path", required=True, type=str, help="Checkpoint path"
    )
    parser.add_argument(
        "--device", type=str, help="Device to use", choices=["cpu", "cuda", "auto"]
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        help="Value for lambda_1 and lambda_2",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size", default=4)
    parser.add_argument(
        "--vectorizer-dim",
        type=int,
        help="Vectorizer dim",
        choices=[512, 2048],
    )
    parser.add_argument("--use-dp-safe", action="store_true", help="Use DP safe")
    parser.add_argument(
        "--model-name",
        required=True,
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
        "--skip-mri-save",
        action="store_true",
        help="Skip saving MRI files and only save vectorized outputs",
    )
    parser.add_argument(
        "--use-custom-checkpoint",
        action="store_true",
        help="Load custom checkpoint with individual HAGAN components",
    )
    # One of "auto", "cpu", "cuda"
    parser.add_argument(
        "--custom-checkpoint-map-location",
        type=str,
        help="Map location for custom checkpoint",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )

    return parser.parse_args()


def get_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    latent_dim: int = 1024,
    lambdas: float = 1.0,
    use_dp_safe: bool = False,
    use_custom_checkpoint: bool = False,
    map_location: str = "auto",
) -> L.LightningModule:
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
    """Load individual components of HAGAN from a custom checkpoint.

    Args:
        checkpoint_path: Path to checkpoint containing individual components

    Returns:
        Tuple containing:
        - generator: Generator network
        - encoder: Encoder network
        - discriminator: Discriminator network
        - code_discriminator: Code discriminator network
    """
    g_state_dict, d_state_dict, e_state_dict, sub_e_state_dict = (
        load_checkpoint_components(checkpoint_path, map_location=map_location)
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


def main():
    print("Generating samples")
    args = pars_args()
    print(vars(args))

    print(
        "devices:",
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )

    device = (
        args.device
        if args.device != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("device:", device)
    vectorizer_depth = 10 if args.vectorizer_dim == 512 else 50
    print("vectorizer depth:", vectorizer_depth)

    # Load model
    print("Loading MRI vectorizer")
    mri_vectorizer = get_mri_vectorizer(vectorizer_depth).eval().to(device)
    print("Loading model")
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
        .to(device)
    )

    print("Generating vectorizer out array")
    mri_vectorizer_out = torch.zeros((args.n_samples, args.vectorizer_dim))

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    outer_bar = tqdm.tqdm(
        batched(range(args.n_samples), args.batch_size),
        total=args.n_samples // args.batch_size,
        desc="Generating samples",
    )

    with torch.no_grad():
        for batch_ids in outer_bar:
            found = False
            # if path below exists for all sample ids in batch ids, skip generation
            if all(
                Path(f"{args.output_dir}/sample_{sample_id}.nii.gz").exists()
                for sample_id in batch_ids
            ):
                # load mri from file:
                samples = [
                    nib.load(f"{args.output_dir}/sample_{sample_id}.nii.gz")  # type: ignore
                    for sample_id in batch_ids
                ]
                samples = [sample.get_fdata() for sample in samples]  # type: ignore
                samples = np.stack(samples)
                samples = np.expand_dims(samples, axis=1)
                samples = samples.astype(np.float32)
                sample = samples
                found = True
            else:
                if args.use_small_model:
                    sample = model.sample_small(len(batch_ids))
                else:
                    sample = model.sample(len(batch_ids))
                sample = sample.detach().cpu().numpy()
                found = False

            inner_bar = tqdm.tqdm(
                enumerate(batch_ids),
                total=len(batch_ids),
                desc="Saving samples",
                leave=False,
                position=1,
            )

            for i, sample_id in inner_bar:
                # Save MRI NIfTI sample
                sample_i = sample[i, 0]
                sample[i, 0] = sample_i

                if not args.skip_mri_save and not found:
                    sample_mri = numpy_to_nifti(sample_i)
                    nib.save(sample_mri, f"{args.output_dir}/sample_{sample_id}.nii.gz")  # type: ignore

                # Get MRI vectorizer output
                mri_vectorizer_out[sample_id] = (
                    mri_vectorizer(
                        torch.from_numpy(sample_i).unsqueeze(0).unsqueeze(0).to(device)
                    )
                    .detach()
                    .cpu()
                    .squeeze(0)
                    .squeeze(0)
                )

        if args.from_authors:
            np.save(f"{args.output_dir}/generated-from-authors.npy", mri_vectorizer_out)
        else:
            model_name = (
                f"hagan-l{int(args.lambdas)}"
                if args.model_name == "hagan"
                else args.model_name
            )
            np.save(f"{args.output_dir}/generated-{model_name}.npy", mri_vectorizer_out)

        print("Finished generating samples")


if __name__ == "__main__":
    main()
