import argparse
from pathlib import Path

import torch

from thesis_code.dataloading.utils import save_mri
from thesis_code.models.gans.hagan.hagan import LitHAGAN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MRI samples from a trained HA-GAN model"
    )
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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.checkpoint_path}")
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
