import argparse
import numpy as np
import torch
from monai.metrics.fid import FIDMetric


def compute_fid(true_file, generated_file):
    true_vec = np.load(true_file)
    generated_vec = np.load(generated_file)

    true_torch_vec = torch.from_numpy(true_vec)
    generated_torch_vec = torch.from_numpy(generated_vec)

    fid = FIDMetric()
    fid_score = fid(true_torch_vec, generated_torch_vec)
    return fid_score


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID score between true and generated distributions"
    )
    parser.add_argument(
        "true_file", type=str, help="Path to the true distribution numpy file"
    )
    parser.add_argument(
        "generated_file", type=str, help="Path to the generated distribution numpy file"
    )
    args = parser.parse_args()

    fid_score = compute_fid(args.true_file, args.generated_file)
    print(f"FID Score: {fid_score}")


if __name__ == "__main__":
    main()
