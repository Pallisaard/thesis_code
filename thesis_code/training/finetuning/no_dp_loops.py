from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from thesis_code.dataloading.mri_dataset import MRIDataset
from thesis_code.models.gans.hagan import (
    compute_d_loss,
    compute_e_loss,
    compute_g_loss,
    compute_sub_e_loss,
    prepare_data,
    save_mri,
    safe_sample,
)
from .utils import checkpoint_no_dp_model
from .types import (
    NoDPOptimizers,
    NoDPDataLoaders,
    NoDPState,
    NoDPModels,
    LossFNs,
    TrainingStats,
)


def setup_no_dp_hagan_training(
    generator: nn.Module,
    discriminator: nn.Module,
    encoder: nn.Module,
    sub_encoder: nn.Module,
    train_dataset: MRIDataset,
    val_dataset: MRIDataset,
    device: str,
    num_workers: int = 4,
    lr_g: float = 0.0001,
    lr_d: float = 0.0004,
    lr_e: float = 0.0001,
    batch_size: int = 4,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    log_every_n_steps: Optional[int] = None,
) -> Tuple[NoDPModels, NoDPOptimizers, NoDPDataLoaders, NoDPState, LossFNs]:
    g_optimizer = optim.Adam(
        generator.parameters(), lr=lr_g, betas=(0.0, 0.999), eps=1e-8
    )
    d_dpoptimizer = optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(0.0, 0.999), eps=1e-8
    )
    e_dpoptimizer = optim.Adam(
        encoder.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8
    )
    sub_e_dpoptimizer = optim.Adam(
        sub_encoder.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8
    )
    no_dp_optimizers = NoDPOptimizers(
        g_opt=g_optimizer,
        d_opt=d_dpoptimizer,
        e_opt=e_dpoptimizer,
        sub_e_opt=sub_e_dpoptimizer,
    )

    # Create the models
    no_dp_models = NoDPModels(
        G=generator, D=discriminator, E=encoder, Sub_E=sub_encoder
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    no_dp_dataloaders = NoDPDataLoaders(train=train_dataloader, val=val_dataloader)

    state = NoDPState(
        lambdas=1.0,
        device=device,
        latent_dim=1024,
        training_stats=TrainingStats(log_every_n_steps=log_every_n_steps),
    )

    loss_fns = LossFNs()

    return no_dp_models, no_dp_optimizers, no_dp_dataloaders, state, loss_fns


def no_dp_training_step(
    models: NoDPModels,
    optimizers: NoDPOptimizers,
    loss_fns: LossFNs,
    state: NoDPState,
    batch: torch.Tensor,
) -> NoDPState:
    generator = models.G
    discriminator = models.D
    encoder = models.E
    sub_encoder = models.Sub_E
    g_optimizer = optimizers.g_opt
    d_optimizer = optimizers.d_opt
    e_optimizer = optimizers.e_opt
    sub_e_optimizer = optimizers.sub_e_opt
    l1_loss = loss_fns.l1
    bce_loss = loss_fns.bce

    data_dict = prepare_data(batch=batch, latent_dim=state.latent_dim)
    real_images = data_dict["real_images"]
    _batch_size = data_dict["batch_size"]
    real_images_small = data_dict["real_images_small"]
    crop_idx = data_dict["crop_idx"]
    real_images_crop = data_dict["real_images_crop"]
    noise = data_dict["noise"]
    real_labels = data_dict["real_labels"]
    fake_labels = data_dict["fake_labels"]

    # Train Discriminator
    generator.requires_grad_(False)
    discriminator.requires_grad_(True)
    encoder.requires_grad_(False)
    sub_encoder.requires_grad_(False)
    d_optimizer.zero_grad()
    d_loss = compute_d_loss(
        D=discriminator,
        G=generator,
        bce_loss=bce_loss,
        real_images_crop=real_images_crop,
        real_images_small=real_images_small,
        crop_idx=crop_idx,
        noise=noise,
        real_labels=real_labels,
        fake_labels=fake_labels,
    )
    d_loss.backward()
    d_optimizer.step()
    state.training_stats.train_metrics.d_loss.append(d_loss.detach().cpu().item())

    # Train Generator
    generator.requires_grad_(True)
    discriminator.requires_grad_(False)
    encoder.requires_grad_(False)
    sub_encoder.requires_grad_(False)
    g_optimizer.zero_grad()
    g_loss = compute_g_loss(
        G=generator,
        D=discriminator,
        bce_loss=bce_loss,
        noise=noise,
        crop_idx=crop_idx,
        real_labels=real_labels,
    )
    g_loss.backward()
    g_optimizer.step()
    state.training_stats.train_metrics.g_loss.append(g_loss.detach().cpu().item())

    # Train Encoder
    generator.requires_grad_(False)
    discriminator.requires_grad_(False)
    encoder.requires_grad_(True)
    sub_encoder.requires_grad_(False)
    e_optimizer.zero_grad()
    e_loss = compute_e_loss(
        E=encoder,
        G=generator,
        l1_loss=l1_loss,
        real_images_crop=real_images_crop,
        lambda_1=state.lambdas,
    )
    e_loss.backward()
    e_optimizer.step()
    state.training_stats.train_metrics.e_loss.append(e_loss.detach().cpu().item())

    # Train Sub-Encoder
    generator.requires_grad_(False)
    discriminator.requires_grad_(False)
    encoder.requires_grad_(False)
    sub_encoder.requires_grad_(True)
    sub_e_optimizer.zero_grad()
    sub_e_loss = compute_sub_e_loss(
        E=encoder,
        Sub_E=sub_encoder,
        G=generator,
        l1_loss=l1_loss,
        real_images=real_images,
        real_images_crop=real_images_crop,
        real_images_small=real_images_small,
        crop_idx=crop_idx,
        lambda_2=state.lambdas,
    )
    sub_e_loss.backward()
    sub_e_optimizer.step()
    state.training_stats.train_metrics.sub_e_loss.append(
        sub_e_loss.detach().cpu().item()
    )

    return state


def no_dp_validation_step(
    models: NoDPModels,
    loss_fns: LossFNs,
    state: NoDPState,
    batch: torch.Tensor,
    save_mris: bool = False,
) -> NoDPState:
    generator = models.G
    discriminator = models.D
    encoder = models.E
    sub_encoder = models.Sub_E
    l1_loss = loss_fns.l1
    bce_loss = loss_fns.bce

    data_dict = prepare_data(batch=batch, latent_dim=state.latent_dim)
    real_images = data_dict["real_images"]
    _batch_size = data_dict["batch_size"]
    real_images_small = data_dict["real_images_small"]
    crop_idx = data_dict["crop_idx"]
    real_images_crop = data_dict["real_images_crop"]
    noise = data_dict["noise"]
    real_labels = data_dict["real_labels"]
    fake_labels = data_dict["fake_labels"]

    val_d_loss = compute_d_loss(
        D=discriminator,
        G=generator,
        bce_loss=bce_loss,
        real_images_crop=real_images_crop,
        real_images_small=real_images_small,
        crop_idx=crop_idx,
        noise=noise,
        real_labels=real_labels,
        fake_labels=fake_labels,
    )
    state.training_stats.val_metrics.val_d_loss.append(val_d_loss.detach().cpu().item())

    val_g_loss = compute_g_loss(
        G=generator,
        D=discriminator,
        bce_loss=bce_loss,
        noise=noise,
        crop_idx=crop_idx,
        real_labels=real_labels,
    )
    state.training_stats.val_metrics.val_g_loss.append(val_g_loss.detach().cpu().item())

    val_e_loss = compute_e_loss(
        E=encoder,
        G=generator,
        l1_loss=l1_loss,
        real_images_crop=real_images_crop,
        lambda_1=state.lambdas,
    )
    state.training_stats.val_metrics.val_e_loss.append(val_e_loss.detach().cpu().item())

    val_sub_e_loss = compute_sub_e_loss(
        E=encoder,
        Sub_E=sub_encoder,
        G=generator,
        l1_loss=l1_loss,
        real_images=real_images,
        real_images_crop=real_images_crop,
        real_images_small=real_images_small,
        crop_idx=crop_idx,
        lambda_2=state.lambdas,
    )
    state.training_stats.val_metrics.val_sub_e_loss.append(
        val_sub_e_loss.detach().cpu().item()
    )

    if save_mris:
        # Save a sample of the generated images
        true_example_save_path = (
            state.training_stats.log_dir
            / f"true_example_{state.training_stats.step}.nii.gz"
        )
        save_mri(real_images, true_example_save_path)

        # Save a sample of the generated images
        fake_images = safe_sample(2, generator, state.latent_dim, device=state.device)
        fake_example_save_path = (
            state.training_stats.log_dir
            / f"fake_example_{state.training_stats.step}.nii.gz"
        )
        save_mri(fake_images, fake_example_save_path)

    return state


def no_dp_training_loop_for_n_steps(
    models: NoDPModels,
    optimizers: NoDPOptimizers,
    dataloaders: NoDPDataLoaders,
    loss_fns: LossFNs,
    state: NoDPState,
    n_steps: int,
    checkpoint_path: str = "dp_training/checkpoints",
) -> NoDPState:
    data_iter = iter(dataloaders.train)
    while state.training_stats.step < n_steps:
        state.training_stats.step += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reinitialize the iterator if the previous one is exhausted
            data_iter = iter(dataloaders.train)
            batch = next(data_iter)

        state.training_stats.step += 1
        state = no_dp_training_step(
            models=models,
            optimizers=optimizers,
            loss_fns=loss_fns,
            state=state,
            batch=batch,
        )

        if (
            state.training_stats.log_every_n_steps is not None
            and state.training_stats.step % state.training_stats.log_every_n_steps == 0
        ):
            for i, val_batch in enumerate(dataloaders.val):
                # val_batch = next(iter(dataloaders.val))
                state = no_dp_validation_step(
                    models=models,
                    loss_fns=loss_fns,
                    state=state,
                    batch=val_batch,
                    save_mris=i == 0,
                )

    checkpoint_no_dp_model(
        models.G,
        state,
        f"{checkpoint_path}/generator_final_steps={n_steps}.pth",
    )

    return state
