import argparse
from typing import Literal

from lightning.pytorch.trainer import Trainer
import lightning as L

# from models import VAE3DLightningModule
from thesis_code.models import LitVAE3D
from thesis_code.models.gans.kwon_gan import LitKwonGan
from thesis_code.dataloading.mri_datamodule import MRIDataModule
from thesis_code.dataloading.transforms import (
    MRITransform,
    Compose,
    Resize,
    ZScoreNormalize,
    RangeNormalize,
)
from thesis_code.training.callbacks.callbacks import (
    get_checkpoint_callback,
    get_summary_callback,
    get_progress_bar_callback,
)

type MODEL_NAME = Literal["cicek_3d_vae_64", "cicek_3d_vae_256", "kwon_gan"]


def get_specific_model(
    model_class, checkpoint_path: str | None = None, **kwargs
) -> L.LightningModule:
    if checkpoint_path is not None:
        return model_class.load_from_checkpoint(checkpoint_path)
    return model_class(**kwargs)


def get_model(
    model_name: MODEL_NAME, latent_dim: int, load_from_checkpoint: str | None
) -> L.LightningModule:
    match model_name:
        case "cicek_3d_vae_256":
            return get_specific_model(
                LitVAE3D,
                in_shape=(1, 256, 256, 256),
                encoder_out_channels_per_block=[16, 32, 64, 128],
                decoder_out_channels_per_block=[128, 64, 32, 16, 1],
                latent_dim=latent_dim,
            )
        case "cicek_3d_vae_64":
            return get_specific_model(
                LitVAE3D,
                in_shape=(1, 64, 64, 64),
                encoder_out_channels_per_block=[16, 32, 64],
                decoder_out_channels_per_block=[64, 32, 16, 1],
                latent_dim=latent_dim,
            )
        case "kwon_gan":
            return get_specific_model(
                LitKwonGan,
                generator=None,
                critic=None,
                code_critic=None,
                encoder=None,
                lambda_grad_policy=10.0,
                n_critic_steps=5,
                lambda_recon=1.0,
            )


def get_datamodule(
    data_dir: str,
    batch_size: int,
    n_workers: int,
    transform: MRITransform,
    size_limit: int | None,
) -> MRIDataModule:
    return MRIDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=n_workers,
        transform=transform,
        size_limit=size_limit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Model arguments.
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["cicek_3d_vae", "kwon_gan"],
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
    # Transforms
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=[],
        help="List of transforms to apply to the data",
        choices=["resize", "z-normalize", "range-normalize"],
    )
    # Resize transform arguments.
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Size to resize images to before passing to model",
    )
    # Z-normalize transform arguments.
    parser.add_argument(
        "--normalize-dir",
        type=str,
        default=None,
        help="Path to directory containing normalization statistics to use",
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
        "--gpus",
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
    parser.add_argument(
        "--max-time",
        type=str,
        default="00:04:00:00",
        help="Maximum time to train for in hours.",
    )
    parser.add_argument(
        "--callbacks",
        type=str,
        nargs="+",
        default=[],
        help="List of callbacks to use during training.",
        choices=["checkpoint", "summary", "progress"],
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="lightning/checkpoints",
        help="Path to save model checkpoints.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="Number of top models to save when checkpointing.",
    )
    parser.add_argument(
        "--save-last",
        type=bool,
        default=True,
        help="Whether to save the last model checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-monitor",
        type=str,
        default="ssim",
        help="Metric to monitor for checkpointing.",
    )
    parser.add_argument(
        "--load-from-checkpoint",
        type=str,
        default=None,
        help="Path to load model from checkpoint. Default None will initialize a new model.",
    )
    return parser.parse_args()


def check_args(args: argparse.Namespace):
    if "resize" in args.transforms and args.resize_size is None:
        raise ValueError("Must provide resize size if using resize transform")
    if "z-normalize" in args.transforms and args.normalize_dir is None:
        raise ValueError(
            "Must provide normalization directory if using z-normalize transform"
        )


def get_transforms(args: argparse.Namespace) -> MRITransform:
    transforms = []
    if "resize" in args.transforms:
        transforms.append(Resize(size=args.resize_size))
    if "z-normalize" in args.transforms:
        zscore_normalize = ZScoreNormalize.load_from_disk(args.normalize_dir)
        transforms.append(zscore_normalize)
    if "range-normalize" in args.transforms:
        transforms.append(RangeNormalize(args.normalize_min, args.normalize_max))
    return Compose(transforms)


def get_callbacks_from_args(args: argparse.Namespace) -> list[L.Callback]:
    callbacks = []
    if "checkpoint" in args.callbacks:
        callbacks.append(
            get_checkpoint_callback(
                path=args.checkpoint_path,
                model_name=args.model_name,
                monitor=args.checkpoint_monitor,
                save_last=args.save_last,
                save_top_k=args.save_top_k,
            )
        )
    if "summary" in args.callbacks:
        callbacks.append(get_summary_callback())
    if "progress" in args.callbacks:
        callbacks.append(get_progress_bar_callback())
    return callbacks


def main():
    print("Running pre-training script")
    args = parse_args()
    check_args(args)

    print("Creating model")
    model = get_model(args.model_name, args.latent_dim, args.load_from_checkpoint)

    print("Model summary:")

    print("Creating datamodule")
    transform = get_transforms(args)
    data_module = get_datamodule(
        args.data_dir,
        args.batch_size,
        args.n_workers,
        transform=transform,
        size_limit=100 if args.fast_dev_run else None,
    )
    print("Model:", model)
    print("Data module:", data_module)

    print("Creating trainer")
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        callbacks=get_callbacks_from_args(args),
    )
    print("Trainer:", trainer)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
