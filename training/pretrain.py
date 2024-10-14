import argparse
from typing import Literal

from lightning.pytorch.trainer import Trainer
import lightning as L

# from models import VAE3DLightningModule
from models import VAE3DLightningModule
from dataloading.mri_datamodule import MRIDataModule
from dataloading.transforms import MRITransform, Compose, Resize, ZScoreNormalize
from dataloading.mri_dataset import MRIDataset

type MODEL_NAME = Literal["cicek_3d_vae"]


def get_transforms(size: int, normalize_dir: str | None = None) -> MRITransform:
    if normalize_dir is not None:
        zscore_normalize = ZScoreNormalize.load_from_disk(normalize_dir)
    else:
        zscore_normalize = ZScoreNormalize.from_parameters(mean=0.0, std=1.0)

    return Compose(
        [
            Resize(size=size),
            zscore_normalize,
        ]
    )


def get_model(model_name: MODEL_NAME, latent_dim: int) -> L.LightningModule:
    match model_name:
        case "cicek_3d_vae":
            return VAE3DLightningModule(
                in_shape=(1, 256, 256, 256),
                encoder_out_channels_per_block=[16, 32, 64, 128],
                decoder_out_channels_per_block=[128, 64, 32, 16, 1],
                latent_dim=latent_dim,
            )


def get_data_module(data_dir: str, batch_size: int, n_workers: int):
    return MRIDataModule(
        data_dir=data_dir, batch_size=batch_size, num_workers=n_workers
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Model arguments.
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["cicek_3d_vae"],
        help="Name of the model to train",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=256, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the data directory"
    )

    # Data module arguments.
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--n-workers", type=int, default=0, help="Number of workers for data loader"
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Size to resize images to before passing to model",
    )
    parser.add_argument(
        "--normalize-dir",
        type=str,
        default=None,
        help="Path to directory containing normalization statistics to use",
    )

    # Lightning arguments.
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Type of accelerator to use. ('cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto')",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Type of strategy to use. use 'auto' to let Lightning decide and otherwise 'ddp'.",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Number of devices to use. Use 'auto' to use all available devices.",
    )
    parser.add_argument(
        "--fast-dev-run",
        type=bool,
        default=False,
        help="Whether to run a fast development run.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train for.",
    )
    return parser.parse_args()


def main():
    print("Running pre-training script")
    args = parse_args()

    print("Creating model and data module")
    model = get_model(args.model_name, args.latent_dim)
    data_module = get_data_module(args.data_dir, args.batch_size, args.n_workers)
    print("Model:", model)
    print("Data module:", data_module)

    print("Creating trainer")
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.max_epochs,
        callbacks=[],
    )
    print("Trainer:", trainer)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
