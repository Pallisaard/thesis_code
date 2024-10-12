### Count number of files in data_dir, use arguments 'split_ratio' or 'val_size' to split data into train and validation sets, and save the split data into two separate directories.'
# The split should be randomly drawn indices and should be reproducible (an argument should be seed).

import argparse

from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Path to directory containing the data.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=None,
        help="Ratio of data to use for training.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Number of samples to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    samples = list(data_dir.glob("scans/*.nii.gz"))
    print(f"Number of samples: {len(samples)}")

    if args.split_ratio is None and args.val_size is None:
        raise ValueError("Either 'split_ratio' or 'val_size' must be provided.")
    elif args.split_ratio is not None and args.val_size is not None:
        raise ValueError("Only one of 'split_ratio' or 'val_size' can be provided.")
    elif args.val_size is not None:
        val_size = args.val_size
    else:
        val_size = int(len(samples) * (1 - args.split_ratio))

    train_size = len(samples) - val_size
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")

    val_indices = np.random.choice(len(samples), size=val_size, replace=False)
    val_samples: list[Path] = [samples[i] for i in val_indices]
    train_samples: list[Path] = [
        sample for i, sample in enumerate(samples) if i not in val_indices
    ]

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    for i, sample in enumerate(samples):
        if i < train_size:
            dest_dir = train_dir
        else:
            dest_dir = val_dir

        dest_file = dest_dir / sample.name
        sample.replace(dest_file)
