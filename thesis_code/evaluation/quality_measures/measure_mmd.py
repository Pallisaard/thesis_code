import argparse

import numpy as np
import torch
from thesis_code.metrics.mmd import MMDMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MMD score between true and generated MRI distributions"
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
        help="Path to save the MMD score",
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
    print("Computing MMD scores")
    print(vars(args))

    # Load vectors
    print("Loading vectors")
    true_vec = torch.from_numpy(np.load(args.true_vectors)).to(args.device)
    generated_vec = torch.from_numpy(np.load(args.generated_vectors)).to(args.device)

    print(f"True vectors shape: {true_vec.shape}")
    print(f"Generated vectors shape: {generated_vec.shape}")

    # Setup and compute MMD
    mmd = MMDMetric().to(args.device)
    mmd_score = mmd(true_vec, generated_vec)

    print(f"MMD Score: {mmd_score:.4f}")

    # Save score
    torch.save(mmd_score, args.output_file)


if __name__ == "__main__":
    main()
