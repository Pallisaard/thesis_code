import argparse
from typing import Any, Literal, Optional

import torch
from lightning.pytorch.trainer import Trainer
import lightning as L

# from models import VAE3DLightningModule
from thesis_code.models import LitKwonGan, LitHAGAN, LitWGANGP, LitAlphaGAN, LitVAE3D
from thesis_code.dataloading.mri_datamodule import MRIDataModule, MRIAllTrainDataModule
from thesis_code.dataloading.transforms import (
    MRITransform,
    Compose,
    Resize,
    RangeNormalize,
    RemovePercentOutliers,
)
from thesis_code.training.callbacks.callbacks import (
    get_checkpoint_callback,
    get_summary_callback,
    get_progress_bar_callback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Model arguments.
    parser.add_argument(
        "--model-name",
        required=True,
        choices=[
            "cicek_3d_vae_64",
            "cicek_3d_vae_256",
            "kwon_gan",
            "wgan_gp",
            "alpha_gan",
            "hagan",
        ],
        help="Name of the model to train",
    )
    parser.add_argument("--latent-dim", type=int, default=1024, help="Dimension of the latent space")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data directory")
    parser.add_argument(
        "--use-all-data-for-training",
        action="store_true",
        help="Use all data for training",
    )

    # HAGAN arguments.
    parser.add_argument("--lambdas", type=float, default=1.0, help="Lambdas for HAGAN")
    # Data module arguments.
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for data loader")
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
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=0.01,
        help="Percentile of outliers to remove from data",
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
        action="store_true",
        help="Whether to run a fast development run.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=-1,
        help="Maximum number of epochs to train for.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum number of steps to train for.",
    )
    parser.add_argument(
        "--max-time",
        type=str,
        default=None,
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
        "--checkpoint-path",
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
        action="store_true",
        help="Whether to save the last model checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-monitor",
        type=str,
        default="val_total_loss",
        help="Metric to monitor for checkpointing.",
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


MODEL_NAME = Literal["cicek_3d_vae_64", "cicek_3d_vae_256", "kwon_gan", "wgan_gp", "alpha_gan", "hagan"]


def get_model(
    model_name: MODEL_NAME,
    latent_dim: int,
    args: argparse.Namespace | dict[str, Any],
) -> L.LightningModule:
    max_steps = args.get("max_steps", -1) if isinstance(args, dict) else args.max_steps
    lambdas = args.get("lambdas", 5.0) if isinstance(args, dict) else args.lambdas

    if model_name == "cicek_3d_vae_256":
        return LitVAE3D(
            in_shape=(1, 256, 256, 256),
            encoder_out_channels_per_block=[8, 16, 32, 64],
            decoder_out_channels_per_block=[64, 64, 16, 8, 1],
            latent_dim=latent_dim,
            beta_annealing="monotonic",
            max_beta=4.0,
            # 100,000 beta annealing steps with batch size = 32 and dataset size 2740
            warmup_epochs=32 * 100000 // 2740,
        )
    elif model_name == "cicek_3d_vae_64":
        return LitVAE3D(
            in_shape=(1, 64, 64, 64),
            encoder_out_channels_per_block=[16, 32, 64],
            decoder_out_channels_per_block=[64, 32, 16, 1],
            latent_dim=latent_dim,
            constant_beta=1.0,
            max_beta=4.0,
            warmup_epochs=32 * int(max_steps * 1 / 2) // 2740,
        )
    elif model_name == "alpha_gan":
        return LitAlphaGAN(latent_dim=latent_dim)
    elif model_name == "wgan_gp":
        return LitWGANGP(
            latent_dim=latent_dim,
            n_critic_steps=10,
            n_generator_steps=1,
            gp_weight=10.0,
        )
    elif model_name == "kwon_gan":
        return LitKwonGan(
            latent_dim=latent_dim,
            n_critic_steps=5,
            lambda_recon=lambdas,
            lambda_gp=lambdas,
        )
    elif model_name == "hagan":
        return LitHAGAN(
            latent_dim=latent_dim,
            lambda_1=lambdas,
            lambda_2=lambdas,
            use_dp_safe=True,
        )
    else:
        raise ValueError(f"Model name {model_name} not recognized")


def get_datamodule(
    data_path: str,
    batch_size: int,
    num_workers: int,
    transform: Optional[MRITransform],
    size_limit: Optional[int],
    use_all_data_for_training: bool = False,
) -> MRIDataModule | MRIAllTrainDataModule:
    if use_all_data_for_training:
        return MRIAllTrainDataModule(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            size_limit=size_limit,
        )
    return MRIDataModule(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        size_limit=size_limit,
    )


def check_args(args: argparse.Namespace):
    if "resize" in args.transforms and args.resize_size is None:
        raise ValueError("Must provide resize size if using resize transform")
    if "z-normalize" in args.transforms and args.normalize_dir is None:
        raise ValueError("Must provide normalization directory if using z-normalize transform")


def get_transforms(args: argparse.Namespace) -> Optional[MRITransform]:
    transforms: list[MRITransform] = []
    for transform in args.transforms:
        if transform == "resize":
            transforms.append(Resize(size=args.resize_size))
        elif transform == "range-normalize":
            transforms.append(RangeNormalize(args.normalize_min, args.normalize_max))
        elif transform == "remove-percent-outliers":
            transforms.append(RemovePercentOutliers(args.outlier_percentile))
        else:
            raise ValueError(f"Transform {transform} not recognized")
    if transforms == []:
        return None
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
    print("Arguments:", vars(args))

    print(
        "devices:",
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )

    print("Creating model")
    model = get_model(args.model_name, args.latent_dim, args)

    print("Creating datamodule")
    transform = get_transforms(args)
    data_module = get_datamodule(
        args.data_path,
        args.batch_size,
        args.num_workers,
        transform=transform,
        size_limit=100 if args.fast_dev_run else None,
        use_all_data_for_training=args.use_all_data_for_training,
    )

    print("Data module:", data_module)

    print("transforms:", transform)

    callbacks = get_callbacks_from_args(args)
    print("callbacks:", callbacks)

    print("Creating trainer")
    trainer = Trainer(
        log_every_n_steps=args.log_every_n_steps,
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        max_time=args.max_time,
        callbacks=callbacks,
        default_root_dir=args.default_root_dir,
        check_val_every_n_epoch=10 if args.model_name in ("wgan_gp", "cicek_3d_vae_64") else 1,
    )

    print("Fitting model")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.load_from_checkpoint)
    # trainer.print(torch.cuda.memory_summary())

    print("Testing model")
    # trainer.test(model, datamodule=data_module)
    # trainer.print(torch.cuda.memory_summary())

    print("Finished pre-training script")


if __name__ == "__main__":
    main()
