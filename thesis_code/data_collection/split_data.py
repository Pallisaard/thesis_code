### Count number of files in data_path, use arguments 'split_ratio' or 'val_size' to split data into train and validation sets, and save the split data into two separate directories.'
# The split should be randomly drawn indices and should be reproducible (an argument should be seed).

import argparse
import shutil

from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to directory containing the data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/",
        help="Path to directory where the split data will be saved.",
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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode. This prints the number of samples, the split sizes and the move commands to be made, but does not move any files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise ValueError(f"Output directory not found: {output_dir}")

    samples = list(data_path.glob("*.nii.gz"))
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

    # Create directories for train and validation data
    train_dir = output_dir / "train_data"
    val_dir = output_dir / "val_data"

    # Check if directories already exist and raise an error if they do
    if train_dir.exists() or val_dir.exists():
        raise FileExistsError(
            "Train or validation directory already exists. Please remove or rename existing directories before running the script."
        )

    # Create new directories
    if not args.test:
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
    else:
        print("Would create directories:")
        print(f"Train: {train_dir}")
        print(f"Validation: {val_dir}")

    print()

    move_fn = shutil.move if not args.test else lambda x, y: print(f"Move {x} -> {y}")

    # Move files according to the random split
    print("begining training data move")
    n_train_examples_moved = 0
    for sample in train_samples:
        dest_file = train_dir / sample.name
        move_fn(str(sample), str(dest_file))
        n_train_examples_moved += 1

    print(f"Moved {n_train_examples_moved} training examples")
    print()

    print("begining validation data move")
    n_val_examples_moved = 0
    for sample in val_samples:
        dest_file = val_dir / sample.name
        move_fn(str(sample), str(dest_file))
        n_val_examples_moved += 1

    print(f"Moved {n_val_examples_moved} validation examples")


if __name__ == "__main__":
    main()
