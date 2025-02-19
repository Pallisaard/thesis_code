import argparse
from pathlib import Path

import nibabel as nib
import torch
import tqdm
import lightning as L

from thesis_code.models.gans import LitHAGAN
from thesis_code.training.utils import numpy_to_nifti


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
        LitHAGAN.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            map_location=device,
            latent_dim=1024,
            lambda_1=args.lambdas,
            lambda_2=args.lambdas,
            use_dp_safe=True,
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
            samples = samples.detach().cpu().numpy()

            # Save each sample in the batch
            for i in range(current_batch_size):
                sample_idx = batch_start + i
                sample = samples[i, 0]  # Get the first channel

                # Convert to NIfTI and save
                sample_mri = numpy_to_nifti(sample)
                output_path = output_dir / f"sample_{sample_idx:04d}.nii.gz"
                nib.save(sample_mri, str(output_path))

    print(f"Successfully generated {args.n_samples} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
