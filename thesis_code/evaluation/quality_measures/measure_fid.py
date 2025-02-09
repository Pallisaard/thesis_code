import argparse

import numpy as np
import torch
from monai.metrics.fid import FIDMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID score between true and generated MRI distributions"
    )
    parser.add_argument(
        "--true-vectors",
        required=True,
        type=str,
        help="Path to the true distribution vectors (.npy file)",
    )
    parser.add_argument(
        "--generated-vectors",
        required=True,
        type=str,
        help="Path to the generated distribution vectors (.npy file)",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=str,
        help="Path to save the FID score",
    )
    parser.add_argument(
        "--device",
        required=True,
        type=str,
        help="Device to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Computing FID scores")
    print(vars(args))

    # Load vectors
    print("Loading vectors")
    true_vec = torch.from_numpy(np.load(args.true_vectors)).to(args.device)
    generated_vec = torch.from_numpy(np.load(args.generated_vectors)).to(args.device)

    print(f"True vectors shape: {true_vec.shape}")
    print(f"Generated vectors shape: {generated_vec.shape}")

    # Setup and compute FID
    fid = FIDMetric().to(args.device)
    fid_score = fid(true_vec, generated_vec)

    print(f"FID Score: {fid_score:.4f}")

    # Save score
    torch.save(fid_score, args.output_file)


if __name__ == "__main__":
    main()
