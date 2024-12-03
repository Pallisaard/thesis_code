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
        default=None,
    )
    parser.add_argument(
        "--make-filename-file",
        action="store_true",
        help="Make a file with the filenames of the samples",
    )

    return parser.parse_args()


def main():
    args = pars_args()

    # Load model
    print("Loading MRI vectorizer")
    mri_vectorizer = get_mri_vectorizer(50).eval().to(args.device)

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    if not Path(args.data_dir).exists():
        raise ValueError(f"Data directory {args.data_dir} does not exist")

    all_niis = list(Path(args.data_dir).rglob("*.nii.gz"))
    test_size = len(all_niis) if args.test_size is None else args.test_size

    if args.make_filename_file:
        with open(f"{args.output_dir}/filenames.txt", "w") as f:
            for nii_path in all_niis:
                f.write(f"{nii_path}\n")

    if len(all_niis) > test_size:
        all_niis = all_niis[:test_size]

    print("Generating vectorizer out array")
    mri_vectorizer_out = torch.zeros((test_size, 2048))

    inner_bar = tqdm.tqdm(
        enumerate(all_niis),
        total=len(all_niis),
        desc="Generating vectors for true data",
    )

    print("Saving samples")
    for i, nii_path in inner_bar:
        mri_i = nib.load(nii_path)  # type: ignore
        data_i = mri_i.get_fdata()  # type: ignore
        data_i = 0.5 * data_i + 0.5

        # Get MRI vectorizer output
        inputs = (
            torch.from_numpy(data_i).float().unsqueeze(0).unsqueeze(0).to(args.device)
        )
        with torch.no_grad():
            mri_vectorizer_out[i] = (
                mri_vectorizer(inputs).detach().cpu().squeeze(0).squeeze(0)
            )

    np.save(f"{args.output_dir}/mri_vectorizer_2048_out.npy", mri_vectorizer_out)


if __name__ == "__main__":
    main()
