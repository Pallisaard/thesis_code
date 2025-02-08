import argparse
import numpy as np
import torch

from thesis_code.metrics.mmd import MMDMetric


def compute_mmd(true_file, generated_file):
    true_vec = np.load(true_file)
    generated_vec = np.load(generated_file)

    true_torch_vec = torch.from_numpy(true_vec)
    generated_torch_vec = torch.from_numpy(generated_vec)

    mmd = MMDMetric()
    mmd_score = mmd(true_torch_vec, generated_torch_vec)
    return mmd_score


def main():
    parser = argparse.ArgumentParser(
        description="Compute MMD score between true and generated distributions"
    )
    parser.add_argument(
        "true_file", type=str, help="Path to the true distribution numpy file"
    )
    parser.add_argument(
        "generated_file", type=str, help="Path to the generated distribution numpy file"
    )
    args = parser.parse_args()

    mmd_score = compute_mmd(args.true_file, args.generated_file)
    print(f"FID Score: {mmd_score}")


if __name__ == "__main__":
    main()
