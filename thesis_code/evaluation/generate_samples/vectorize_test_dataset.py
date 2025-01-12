import argparse
from pathlib import Path

import nibabel as nib
import torch
import numpy as np
import tqdm

from thesis_code.dataloading.transforms import normalize_to
from thesis_code.metrics.utils import get_mri_vectorizer
from thesis_code.dataloading.transforms import Resize


def pars_args():
    parser = argparse.ArgumentParser(description="Vectorize test dataset. Assumes test")

    parser.add_argument("--data-dir", required=True, type=str, help="Data directory")
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, help="Device to use", choices=["cpu", "cuda", "auto"]
    )
    parser.add_argument(
        "--test-size",
        type=int,
        help="Number of samples in the test dataset.",
    )
    parser.add_argument(
        "--use-small-model", action="store_true", help="Use sub generator of HAGAN"
    )
    parser.add_argument(
        "--make-filename-file",
        action="store_true",
        help="Make a file with the filenames of the samples",
    )
    parser.add_argument(
        "--vectorizer-dim",
        type=int,
        help="Vectorizer dim",
        choices=[512, 2048],
    )

    return parser.parse_args()


def main():
    args = pars_args()
    device = (
        args.device
        if args.device != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    vectorizer_depth = 10 if args.vectorizer_dim == 512 else 50

    # Load model
    print("Loading MRI vectorizer")
    mri_vectorizer = get_mri_vectorizer(vectorizer_depth).eval().to(device)

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    if not Path(args.data_dir).exists():
        raise ValueError(f"Data directory {args.data_dir} does not exist")

    all_niis = list(Path(args.data_dir).rglob("*.nii.gz"))
    test_size: int = len(all_niis) if args.test_size is None else args.test_size

    if args.make_filename_file:
        with open(f"{args.output_dir}/filenames.txt", "w") as f:
            for nii_path in all_niis:
                f.write(f"{nii_path}\n")

    if len(all_niis) > test_size:
        all_niis = all_niis[:test_size]

    print("Generating vectorizer out array")
    mri_vectorizer_out = torch.zeros((test_size, args.vectorizer_dim))

    inner_bar = tqdm.tqdm(
        enumerate(all_niis),
        total=len(all_niis),
        desc="Generating vectors for true data",
    )

    transform_resize = Resize(64)

    print("Saving samples")
    for i, nii_path in inner_bar:
        mri_i = nib.load(nii_path)  # type: ignore
        data_i: np.ndarray = mri_i.get_fdata()  # type: ignore
        data_i = normalize_to(data_i, -1, 1)

        # Get MRI vectorizer output
        inputs = torch.from_numpy(data_i).float().unsqueeze(0).to(device)
        if args.use_small_model:
            inputs = transform_resize(inputs).unsqueeze(0)

        with torch.no_grad():
            mri_vectorizer_out[i] = (
                mri_vectorizer(inputs).detach().cpu().squeeze(0).squeeze(0)
            )

    out_vectorizer_name = "true-from-all-dataset.npy"
    out_vectorizer_name: str = (
        out_vectorizer_name + ".npy" if not out_vectorizer_name.endswith(".npy") else ""
    )

    np.save(str(Path(args.output_dir) / out_vectorizer_name), mri_vectorizer_out)


if __name__ == "__main__":
    main()
