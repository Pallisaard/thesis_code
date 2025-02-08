from typing import Optional, Tuple

from opacus.accountants import RDPAccountant
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.data_loader import DPDataLoader
from opacus.validators.module_validator import ModuleValidator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from jax.tree_util import tree_map
from tqdm import tqdm

from thesis_code.dataloading.mri_dataset import MRIDataset
from thesis_code.models.gans.hagan import (
    compute_d_loss,
    compute_e_loss,
    compute_g_loss,
    compute_sub_e_loss,
    prepare_data,
    save_mri,
)
from .utils import checkpoint_dp_model
from .types import (
    DPOptimizers,
    DPDataLoaders,
    DPState,
    DPModels,
    LossFNs,
    TrainingStats,
)


def setup_dp_training(
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
    val_every_n_steps: Optional[int] = None,
    checkpoint_every_n_steps: Optional[int] = None,
    alphas: list[float] = [1.1, 2, 3, 5, 10, 20, 50, 100],
) -> Tuple[DPModels, DPOptimizers, DPDataLoaders, DPState, LossFNs]:
    # Module validation
    fix_and_validate = ModuleValidator.fix_and_validate
    discriminator = GradSampleModule(fix_and_validate(discriminator))
    encoder = GradSampleModule(fix_and_validate(encoder))
    sub_encoder = GradSampleModule(fix_and_validate(sub_encoder))

    dp_models = DPModels(G=generator, D=discriminator, E=encoder, Sub_E=sub_encoder)

    # Optimizer validation
    g_optimizer = optim.Adam(
        generator.parameters(), lr=lr_g, betas=(0.0, 0.999), eps=1e-8
    )
    d_dpoptimizer = DPOptimizer(
        optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.999), eps=1e-8),
        expected_batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    e_dpoptimizer = DPOptimizer(
        optim.Adam(encoder.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8),
        expected_batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    sub_e_dpoptimizer = DPOptimizer(
        optim.Adam(sub_encoder.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8),
        expected_batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    dp_optimizers = DPOptimizers(
        g_opt=g_optimizer,
        d_opt=d_dpoptimizer,
        e_opt=e_dpoptimizer,
        sub_e_opt=sub_e_dpoptimizer,
    )

    sample_rate = batch_size / len(train_dataset)
    train_dataloader = DPDataLoader(
        train_dataset,
        sample_rate=sample_rate,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    dp_dataloaders = DPDataLoaders(train=train_dataloader, val=val_dataloader)

    accountant = RDPAccountant()

    state = DPState(
        privacy_accountant=accountant,
        delta=delta,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        sample_rate=sample_rate,
        alphas=alphas,
        lambdas=1.0,
        device=device,
        latent_dim=1024,
        training_stats=TrainingStats(
            val_every_n_steps=val_every_n_steps,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
        ),
    )

    loss_fns = LossFNs()

    return dp_models, dp_optimizers, dp_dataloaders, state, loss_fns


def dp_training_step(
    models: DPModels,
    optimizers: DPOptimizers,
    loss_fns: LossFNs,
    state: DPState,
    batch: torch.Tensor,
) -> DPState:
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

    if isinstance(batch, list):
        if not batch or batch[0].size(0) < 1:
            return state
        batch_size = batch[0].size(0)
    else:
        if batch.size(0) < 1:
            return state
        batch_size = batch.size(0)

    # print("batch_size:", len(batch))
    # if isinstance(batch, list):
    #     print(batch)

    data_dict = tree_map(
        lambda x: x.to(state.device) if isinstance(x, torch.Tensor) else x,
        prepare_data(batch=batch, latent_dim=state.latent_dim),
    )  # Send to device
    real_images = data_dict["real_images"]
    _batch_size = data_dict["batch_size"]
    real_images_small = data_dict["real_images_small"]
    crop_idx = data_dict["crop_idx"]
    real_images_crop = data_dict["real_images_crop"]
    noise = data_dict["noise"]
    real_labels = data_dict["real_labels"]
    fake_labels = data_dict["fake_labels"]

    # Train Discriminator (D^H, D^L)
    with torch.set_grad_enabled(False):
        generator.eval()
        encoder.eval()
        sub_encoder.eval()

    discriminator.train()
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
    d_loss_metric = d_loss.detach().cpu().item()

    # Train Generator
    with torch.set_grad_enabled(False):
        discriminator.eval()
        encoder.eval()
        sub_encoder.eval()

    generator.train()
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
    g_loss_metric = g_loss.detach().cpu().item()

    # Train Encoder
    with torch.set_grad_enabled(False):
        discriminator.eval()
        encoder.eval()
        sub_encoder.eval()

    generator.train()
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
    e_loss_metric = e_loss.detach().cpu().item()

    # Train Sub-Encoder
    with torch.set_grad_enabled(False):
        discriminator.eval()
        encoder.eval()
        sub_encoder.eval()

    generator.train()
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
    sub_e_loss_metric = sub_e_loss.detach().cpu().item()

    state.training_stats.train_metrics.g_loss.append(g_loss_metric)
    state.training_stats.train_metrics.d_loss.append(d_loss_metric)
    state.training_stats.train_metrics.e_loss.append(e_loss_metric)
    state.training_stats.train_metrics.sub_e_loss.append(sub_e_loss_metric)
    total_loss_metric = (
        d_loss_metric + g_loss_metric + e_loss_metric + sub_e_loss_metric
    )
    state.training_stats.train_metrics.total_loss.append(total_loss_metric)

    # Move privacy accounting here after successful training
    state.privacy_accountant.step(
        noise_multiplier=state.noise_multiplier, sample_rate=state.sample_rate
    )
    state.training_stats.train_metrics.epsilon.append(
        state.privacy_accountant.get_epsilon(state.delta)
    )
    state.training_stats.step += 1

    return state


def dp_validation_step(
    models: DPModels,
    loss_fns: LossFNs,
    state: DPState,
    batch: torch.Tensor,
    save_mris: bool = False,
) -> DPState:
    generator = models.G
    discriminator = models.D
    encoder = models.E
    sub_encoder = models.Sub_E
    l1_loss = loss_fns.l1
    bce_loss = loss_fns.bce

    data_dict = tree_map(
        lambda x: x.to(state.device) if isinstance(x, torch.Tensor) else x,
        prepare_data(batch=batch, latent_dim=state.latent_dim),
    )  # Send to device

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
    val_d_loss_metric = val_d_loss.detach().cpu().item()

    val_g_loss = compute_g_loss(
        G=generator,
        D=discriminator,
        bce_loss=bce_loss,
        noise=noise,
        crop_idx=crop_idx,
        real_labels=real_labels,
    )
    val_g_loss_metric = val_g_loss.detach().cpu().item()

    val_e_loss = compute_e_loss(
        E=encoder,
        G=generator,
        l1_loss=l1_loss,
        real_images_crop=real_images_crop,
        lambda_1=state.lambdas,
    )
    val_e_loss_metric = val_e_loss.detach().cpu().item()

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
    val_sub_e_loss_metric = val_sub_e_loss.detach().cpu().item()

    state.training_stats.val_metrics.g_loss.append(val_g_loss_metric)
    state.training_stats.val_metrics.d_loss.append(val_d_loss_metric)
    state.training_stats.val_metrics.e_loss.append(val_e_loss_metric)
    state.training_stats.val_metrics.sub_e_loss.append(val_sub_e_loss_metric)
    val_total_loss_metric = (
        val_g_loss_metric
        + val_d_loss_metric
        + val_e_loss_metric
        + val_sub_e_loss_metric
    )
    state.training_stats.val_metrics.total_loss.append(val_total_loss_metric)

    if save_mris:
        # Save a sample of the generated images
        true_example_save_path = (
            state.training_stats.log_dir
            / f"true_example_{state.training_stats.step}.nii.gz"
        )
        save_mri(real_images, true_example_save_path)

        # Save a sample of the generated images
        fake_images = generator.sample(2).to(state.device)
        fake_example_save_path = (
            state.training_stats.log_dir
            / f"fake_example_{state.training_stats.step}.nii.gz"
        )
        save_mri(fake_images, fake_example_save_path)

    return state


def training_loop_until_epsilon(
    models: DPModels,
    optimizers: DPOptimizers,
    dataloaders: DPDataLoaders,
    loss_fns: LossFNs,
    state: DPState,
    max_epsilon: float,
    alphas: Optional[list[float]] = None,
    checkpoint_path: str = "dp_training/checkpoints",
) -> DPState:
    state.training_stats.current_epsilon = state.privacy_accountant.get_epsilon(
        state.delta, alphas=alphas
    )

    data_iter = iter(dataloaders.train)
    epoch = 0

    with tqdm(desc="DP training progress.", dynamic_ncols=True, leave=True) as pbar:
        while state.training_stats.current_epsilon < max_epsilon:
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                print(f"\nStarting epoch {epoch}")
                data_iter = iter(dataloaders.train)
                batch = next(data_iter)

            try:
                state = dp_training_step(
                    models=models,
                    optimizers=optimizers,
                    loss_fns=loss_fns,
                    state=state,
                    batch=batch,
                )
            except RuntimeError as e:
                print(f"Error during training step: {e}")
                continue

            # Validation logic
            step = state.training_stats.step
            val_every_n_steps = state.training_stats.val_every_n_steps
            if val_every_n_steps is not None and step % val_every_n_steps == 0:
                for i, val_batch in enumerate(dataloaders.val):
                    state = dp_validation_step(
                        models=models,
                        loss_fns=loss_fns,
                        state=state,
                        batch=val_batch,
                        save_mris=i == 0,  # Only save MRIs for first batch
                    )

            if (
                state.training_stats.checkpoint_every_n_steps is not None
                and step % state.training_stats.checkpoint_every_n_steps == 0
            ):
                checkpoint_dp_model(
                    models,
                    state,
                    f"{checkpoint_path}/dp_model_step={state.training_stats.step}.pth",
                )

            state.training_stats.current_epsilon = state.privacy_accountant.get_epsilon(
                state.delta, alphas=alphas
            )

            pbar.set_postfix_str(
                f"Current epsilon: {state.training_stats.current_epsilon}"
            )
            pbar.update(1)

    return state
