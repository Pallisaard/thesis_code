import argparse
from pathlib import Path

import torch
import tqdm

from thesis_code.training.utils import numpy_to_nifti
from thesis_code.dataloading.utils import save_mri
from thesis_code.evaluation.generate_samples.generate_n_sampled_mris import (
    get_model_from_checkpoint,
)


def get_device(device_arg: str) -> str:
    """Determine the appropriate device to use.

    Args:
        device_arg: Device argument from command line

    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu')
    """
    if device_arg != "auto":
        return device_arg

    # Check for CUDA first
    if torch.cuda.is_available():
        return "cuda"
    # Then check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        return "mps"
    # Fall back to CPU
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MRI samples from a HAGAN model"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--n-samples", required=True, type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "--checkpoint-path", required=True, type=str, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        default=5.0,
        help="Value for lambda_1 and lambda_2 (default: 5.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use for computation (default: auto)",
    )
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
        "--use-custom-checkpoint",
        action="store_true",
        help="Load custom checkpoint with individual HAGAN components",
    )
    parser.add_argument(
        "--use-dp-safe",
        action="store_true",
        help="Use DP safe version of the model",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("Arguments:", vars(args))

    # Set device with enhanced support
    device = get_device(args.device)
    print("Using device:", device)

    if device == "cuda":
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
    elif device == "mps":
        print("Using Apple Silicon GPU")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model from checkpoint:", args.checkpoint_path)
    model = (
        get_model_from_checkpoint(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            latent_dim=1024,
            lambdas=args.lambdas,
            use_dp_safe=args.use_dp_safe,
            use_custom_checkpoint=args.use_custom_checkpoint,
            map_location=device,
        )
        .eval()
        .to(device)
    )

    # Generate samples in batches
    print(f"Generating {args.n_samples} samples...")
    with torch.no_grad():
        for batch_start in tqdm.tqdm(range(0, args.n_samples, args.batch_size)):
            # Calculate batch size (might be smaller for last batch)
            current_batch_size = min(args.batch_size, args.n_samples - batch_start)

            # Generate samples
            samples = model.sample(current_batch_size)
            # Move to CPU before converting to numpy
            samples = samples.detach().cpu()

            # Save each sample in the batch

            sample_idx = batch_start

            output_path = output_dir / f"sample_{sample_idx:04d}.nii.gz"
            save_mri(samples, output_path)

    print(f"Successfully generated {args.n_samples} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
