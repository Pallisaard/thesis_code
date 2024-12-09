import argparse
from pathlib import Path
from typing import Optional

import torch
import lightning as L

from thesis_code.dataloading.mri_dataset import MRIDataset
from thesis_code.models.gans.hagan import LitHAGAN
from . import dp_loops
from . import no_dp_loops
from .utils import checkpoint_dp_model


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
    parser.add_argument(
        "--use-dp", action="store_true", help="Use differential privacy"
    )
    # HAGAN arguments.
    parser.add_argument("--lambdas", type=float, default=1.0, help="Lambdas for HAGAN")
    # Data module arguments.
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers for data loader"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="What device to use. Default is cuda if available, else cpu.",
    )
    parser.add_argument(
        "--max-epsilon",
        type=float,
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
        "--checkpoint-every-n-steps",
        type=int,
        default=1000,
        help="Checkpoint every n steps during training.",
    )
    parser.add_argument(
        "--val-every-n-steps",
        type=int,
        default=25,
        help="Log every n steps during training.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",  # Accepts one or more arguments
        type=float,  # Converts each argument to a float
        default=[1.1, 2, 3, 5, 10, 20, 50, 100],  # Default values if none are provided
        help="List of alpha values for Rényi DP accounting (e.g., 1.1 2 3 5 10 20 50 100)",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="Noise multiplier for DP-SGD",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for DP-SGD",
    )
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for DP-SGD")
    parser.add_argument(
        "--size_limit", type=int, default=None, help="Limit the size of the dataset"
    )
    return parser.parse_args()


def get_datasets(
    data_path: str,
    size_limit: Optional[int],
    use_all_dat_for_training: bool,
):
    if use_all_dat_for_training:
        train_dataset = MRIDataset(data_path, transform=None, size_limit=size_limit)
    else:
        train_dataset = MRIDataset(
            str(Path(data_path) / "train"), transform=None, size_limit=size_limit
        )

    val_dataset = MRIDataset(
        str(Path(data_path) / "val"), transform=None, size_limit=size_limit
    )

    return train_dataset, val_dataset


def get_model(
    latent_dim: int, lambdas: float, load_from_checkpoint: Optional[str]
) -> L.LightningModule:
    if load_from_checkpoint is not None:
        return LitHAGAN.load_from_checkpoint(
            load_from_checkpoint,
            latent_dim=latent_dim,
            lambda_1=lambdas,
            lambda_2=lambdas,
        )
    return LitHAGAN(latent_dim=latent_dim, lambda_1=lambdas, lambda_2=lambdas)


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    """Checks the arguments for consistency and raises an error if they are not. returns the args without modifications."""
    if args.use_dp and args.max_epsilon is None:
        raise ValueError(
            "If using differential privacy, you must set a maximum epsilon value."
        )
    if args.max_epsilon is None and args.max_steps == -1:
        raise ValueError(
            "You must set either --max-epsilon or --max-steps. If both are set, --max-epsilon will be used."
        )
    if args.use_dp and args.max_steps != -1:
        print(
            "Warning: --max-steps will be ignored since --max-epsilon is set. Training will continue until the maximum epsilon is reached."
        )

    return args


def main():
    print("Running fine-tuning script.")
    args = check_args(parse_args())

    # Create checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    while checkpoint_path.exists():
        checkpoint_path = checkpoint_path.with_name(checkpoint_path.name + "_1")
    print("Checkpoint path:", checkpoint_path)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Setting device
    device = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Creating model")
    model = get_model(args.latent_dim, args.lambdas, args.load_from_checkpoint)
    generator = model.G.to(device)
    discriminator = model.D.to(device)
    encoder = model.E.to(device)
    sub_encoder = model.Sub_E.to(device)

    print(
        "devices:",
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )

    train_ds, val_ds = get_datasets(
        args.data_path, args.size_limit, args.use_all_data_for_training
    )
    print("training dataset size:", len(train_ds))
    print("validation dataset size:", len(val_ds))

    if args.max_epsilon is not None:
        print("Will train until epsilon is", args.max_epsilon)
    else:
        print("Will train for", args.max_steps, "steps")

    if args.use_dp:
        print("Setting up DP training")
        models, optimizers, dataloaders, state, loss_fns = dp_loops.setup_dp_training(
            generator=generator,
            discriminator=discriminator,
            encoder=encoder,
            sub_encoder=sub_encoder,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            delta=args.delta,
        )

        print("Starting DP training")
        state = dp_loops.training_loop_until_epsilon(
            models=models,
            optimizers=optimizers,
            dataloaders=dataloaders,
            state=state,
            loss_fns=loss_fns,
            max_epsilon=args.max_epsilon,
            checkpoint_path=args.checkpoint_path,
        )

        print(f"Final epsilon: {state.training_stats.current_epsilon}")

        print("Final checkpoint")
        checkpoint_dp_model(
            models,
            state,
            f"{checkpoint_path}/dp_model_final_epsilon={state.training_stats.current_epsilon}.pth",
        )

    else:
        print("Setting up non-DP training")
        models, optimizers, dataloaders, state, loss_fns = (
            no_dp_loops.setup_no_dp_training(
                generator=generator,
                discriminator=discriminator,
                encoder=encoder,
                sub_encoder=sub_encoder,
                train_dataset=train_ds,
                val_dataset=val_ds,
                device=device,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            )
        )

        if args.max_epsilon is not None:
            print("Starting DP training")
            state = no_dp_loops.training_loop_until_epsilon(
                models=models,
                optimizers=optimizers,
                dataloaders=dataloaders,
                state=state,
                loss_fns=loss_fns,
                max_epsilon=args.max_epsilon,
                checkpoint_path=args.checkpoint_path,
            )

            print(f"Final epsilon: {state.training_stats.current_epsilon}")
        else:
            print("Starting non-DP training")
            no_dp_loops.no_dp_training_loop_for_n_steps(
                models=models,
                optimizers=optimizers,
                dataloaders=dataloaders,
                state=state,
                loss_fns=loss_fns,
                n_steps=args.max_steps,
                checkpoint_path=args.checkpoint_path,
            )

        print("Final checkpoint")
        checkpoint_dp_model(
            models,
            state,
            f"{checkpoint_path}/no_dp_final_epsilon={state.training_stats.current_epsilon}.pth",
        )


if __name__ == "__main__":
    main()
