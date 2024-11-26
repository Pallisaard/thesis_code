import argparse
from pathlib import Path

import nibabel as nib
import torch
import numpy as np
import tqdm

from thesis_code.metrics.utils import get_mri_vectorizer


def pars_args():
    parser = argparse.ArgumentParser(description="Vectorize test dataset. Assumes test")

    parser.add_argument("--data-dir", required=True, type=str, help="Data directory")
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument("--device", type=str, help="Device to use", default="cpu")
    parser.add_argument(
        "--test-size",
        type=int,
        help="Number of samples in the test dataset.",
        default=200,
    )

    return parser.parse_args()


def main():
    args = pars_args()

    # Load model
    print("Loading MRI vectorizer")
    mri_vectorizer = get_mri_vectorizer(10).eval().to(args.device)

    print("Generating vectorizer out array")
    mri_vectorizer_out = torch.zeros((args.test_size, 512))

    all_niis = list(Path(args.data_dir).glob("*.nii.gz"))

    inner_bar = tqdm.tqdm(enumerate(all_niis), desc="Generating vectors for true data")

    print("Saving samples")
    for i, nii_path in inner_bar:
        mri_i = nib.load(nii_path)  # type: ignore
        data_i = mri_i.get_fdata()  # type: ignore

        # Get MRI vectorizer output
        mri_vectorizer_out[i] = (
            mri_vectorizer(
                torch.from_numpy(data_i).unsqueeze(0).unsqueeze(0).to(args.device)
            )
            .detach()
            .cpu()
            .squeeze(0)
            .squeeze(0)
        )

    np.save(f"{args.output_dir}/mri_vectorizer_out.npy", mri_vectorizer_out)


if __name__ == "__main__":
    main()
