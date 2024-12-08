import argparse
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from thesis_code.dataloading.mri_dataset import MRIDataset
from thesis_code.dataloading.transforms import MRITransform
from .types import DPDataLoaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Model arguments.
    parser.add_argument(
        "--latent-dim", type=int, default=1024, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--use-all-data-for-training",
        action="store_true",
        help="Use all data for training",
    )

    # HAGAN arguments.
    parser.add_argument("--lambdas", type=float, default=1.0, help="Lambdas for HAGAN")
    # Data module arguments.
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--strip-skulls",
        action="store_true",
        help="Whether to strip skulls from MRI images during preprocessing",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers for data loader"
    )
    # Transforms
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=[],
        help="List of transforms to apply to the data",
        choices=["resize", "z-normalize", "range-normalize", "remove-percent-outliers"],
    )
    # Resize transform arguments.
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Size to resize images to before passing to model",
    )
    # Range-normalize transform arguments.
    parser.add_argument(
        "--normalize-min",
        type=int,
        default=-1,
        help="If using RangeNormalize transform, the minimum of the normalization range.",
    )
    parser.add_argument(
        "--normalize-max",
        type=int,
        default=1,
        help="If using RangeNormalize transform, the maximum of the normalization range.",
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=0.01,
        help="Percentile of outliers to remove from data",
    )
    # Lightning arguments.
    parser.add_argument(
        "--devices",
        default="auto",
        help="Number of devices to use. Use 'auto' to use all available devices.",
    )
    parser.add_argument(
        "--max-epsilon",
        type=int,
        default=None,
        help="Train until we achieve a maximum epsilon measured by RDP in Opacus. Will override --max-steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum number of steps to train for. Will only be used if --max-epsilon is not set.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="lightning/checkpoints",
        help="Path to save model checkpoints.",
    )
    parser.add_argument(
        "--load-from-checkpoint",
        type=str,
        default=None,
        help="Path to load model from checkpoint. Default None will initialize a new model.",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        default=None,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=25,
        help="Log every n steps during training.",
    )
    return parser.parse_args()


def get_datasets(
    data_path: str,
    transform: Optional[MRITransform],
    size_limit: Optional[int],
    use_all_dat_for_training: bool,
):
    if use_all_dat_for_training:
        train_dataset = MRIDataset(
            data_path, transform=transform, size_limit=size_limit
        )
    else:
        train_dataset = MRIDataset(
            str(Path(data_path) / "train"), transform=transform, size_limit=size_limit
        )

    val_dataset = MRIDataset(
        str(Path(data_path) / "val"), transform=transform, size_limit=size_limit
    )

    return train_dataset, val_dataset
