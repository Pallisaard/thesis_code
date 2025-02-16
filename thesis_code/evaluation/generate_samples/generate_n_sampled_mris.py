import argparse
from itertools import batched
from pathlib import Path

import nibabel as nib
import torch
import numpy as np
import tqdm
import lightning as L

from thesis_code.metrics.utils import get_mri_vectorizer
from thesis_code.models.gans import LitHAGAN
from thesis_code.training.utils import numpy_to_nifti
from thesis_code.models import LitKwonGan, LitWGANGP, LitAlphaGAN, LitVAE3D


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

    return parser.parse_args()


def get_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    latent_dim: int = 1024,
    lambdas: float = 1.0,
    use_dp_safe: bool = False,
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
        return LitHAGAN.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            latent_dim=latent_dim,
            lambda_1=lambdas,
            lambda_2=lambdas,
            use_dp_safe=use_dp_safe,
        )
    else:
        raise ValueError(f"Model name {model_name} not recognized")


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
            if args.use_small_model:
                sample = model.sample_small(len(batch_ids))
            else:
                sample = model.sample(len(batch_ids))
            sample = sample.detach().cpu().numpy()

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

                if not args.skip_mri_save:
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
