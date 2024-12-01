from typing import Optional, Tuple, Union
from dataclasses import dataclass

from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer
from opacus.data_loader import DPDataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from thesis_code.dataloading.mri_dataset import MRIDataset
from thesis_code.models.gans.hagan.hagan import (
    prepare_data,
    compute_d_loss,
    compute_e_loss,
    compute_g_loss,
    compute_sub_e_loss,
)


@dataclass
class DPModels:
    G: nn.Module
    D: GradSampleModule
    E: GradSampleModule
    Sub_E: GradSampleModule


@dataclass
class DPOptimizers:
    g_opt: optim.Optimizer
    d_opt: DPOptimizer
    e_opt: DPOptimizer
    sub_e_opt: DPOptimizer


@dataclass
class DPDataLoaders:
    train: DPDataLoader
    val: DataLoader


@dataclass
class LossFNs:
    l1: nn.L1Loss
    bce: nn.BCELoss


@dataclass
class Metrics:
    d_loss: list[float]
    g_loss: list[float]
    e_loss: list[float]
    sub_e_loss: list[float]
    epsilon: list[float]
    val_d_loss: list[float]
    val_g_loss: list[float]
    val_e_loss: list[float]
    val_sub_e_loss: list[float]

    @staticmethod
    def init() -> "Metrics":
        return Metrics([], [], [], [], [], [], [], [], [])


@dataclass
class DPState:
    privacy_accountant: RDPAccountant
    noise_multiplier: float
    sample_rate: float
    epoch: int
    step: int
    batch_idx: int
    optimizer_idx: int
    delta: float
    latent_dim: int
    lambda_1: float
    lambda_2: float
    metrics: Metrics

    @staticmethod
    def init(
        accountant: RDPAccountant,
        delta: float,
        noise_multiplier: float,
        sample_rate: float,
    ) -> "DPState":
        return DPState(
            privacy_accountant=accountant,
            metrics=Metrics.init(),
            step=0,
            epoch=0,
            batch_idx=0,
            delta=delta,
            optimizer_idx=0,
            latent_dim=1024,
            lambda_1=1.0,
            lambda_2=1.0,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )


def wrap_dpoptimizer(
    optimizer: optim.Optimizer,
    expected_batch_size: int,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
) -> DPOptimizer:
    return DPOptimizer(
        optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
    )


def setup_dp_hagan_training(
    generator: nn.Module,
    discriminator: nn.Module,
    encoder: nn.Module,
    sub_encoder: nn.Module,
    train_dataset: MRIDataset,
    val_dataset: MRIDataset,
    lr_g: float = 0.0001,
    lr_d: float = 0.0004,
    lr_e: float = 0.0001,
    batch_size: int = 4,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
) -> Tuple[DPModels, DPOptimizers, DPDataLoaders, DPState]:
    g_optimizer = optim.Adam(
        generator.parameters(), lr=lr_g, betas=(0.0, 0.999), eps=1e-8
    )
    d_dpoptimizer = DPOptimizer(
        optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.999), eps=1e-8),
        expected_batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    e_dpoptimizer = wrap_dpoptimizer(
        optim.Adam(encoder.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8),
        expected_batch_size=batch_size,
    )
    sub_e_dpoptimizer = wrap_dpoptimizer(
        optim.Adam(sub_encoder.parameters(), lr=lr_e, betas=(0.0, 0.999), eps=1e-8),
        expected_batch_size=batch_size,
    )

    dpoptimizers = DPOptimizers(
        g_opt=g_optimizer,
        d_opt=d_dpoptimizer,
        e_opt=e_dpoptimizer,
        sub_e_opt=sub_e_dpoptimizer,
    )

    discriminator = GradSampleModule(discriminator)
    encoder = GradSampleModule(encoder)
    sub_encoder = GradSampleModule(sub_encoder)

    dpmodels = DPModels(G=generator, D=discriminator, E=encoder, Sub_E=sub_encoder)

    sample_rate = len(train_dataset) / batch_size
    train_dataloader = DPDataLoader(
        train_dataset,
        sample_rate=sample_rate,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dpdataloaders = DPDataLoaders(train=train_dataloader, val=val_dataloader)

    accountant = RDPAccountant()

    state = DPState(
        privacy_accountant=accountant,
        metrics=Metrics.init(),
        step=0,
        epoch=0,
        batch_idx=0,
        delta=delta,
        optimizer_idx=0,
        latent_dim=1024,
        lambda_1=1.0,
        lambda_2=1.0,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
    )

    return dpmodels, dpoptimizers, dpdataloaders, state


def training_step(
    models: DPModels,
    optimizers: DPOptimizers,
    dataloaders: DPDataLoaders,
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
    train_dataloader = dataloaders.train
    val_dataloader = dataloaders.val
    l1_loss = loss_fns.l1
    bce_loss = loss_fns.bce

    data_dict = prepare_data(batch=batch, latent_dim=state.latent_dim)
    real_images = data_dict["real_images"]
    batch_size = data_dict["batch_size"]
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
    state.metrics.d_loss.append(d_loss.detach().cpu().item())

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
    state.metrics.g_loss.append(g_loss.detach().cpu().item())

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
        lambda_1=state.lambda_1,
    )
    e_loss.backward()
    e_optimizer.step()
    state.metrics.e_loss.append(e_loss.detach().cpu().item())

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
        lambda_2=state.lambda_2,
    )
    sub_e_loss.backward()
    sub_e_optimizer.step()
    state.metrics.sub_e_loss.append(sub_e_loss.detach().cpu().item())

    state.metrics.epsilon.append(state.privacy_accountant.get_epsilon(state.delta))

    state.privacy_accountant.step(
        noise_multiplier=state.noise_multiplier, sample_rate=state.sample_rate
    )

    return state


def validation_step(
    models: DPModels,
    dataloaders: DPDataLoaders,
    loss_fns: LossFNs,
    state: DPState,
) -> DPState:
    generator = models.G
    discriminator = models.D
    encoder = models.E
    sub_encoder = models.Sub_E
    val_dataloader = dataloaders.val
    l1_loss = loss_fns.l1
    bce_loss = loss_fns.bce

    for batch in val_dataloader:
        data_dict = prepare_data(batch=batch, latent_dim=state.latent_dim)
        real_images = data_dict["real_images"]
        batch_size = data_dict["batch_size"]
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
        state.metrics.val_d_loss.append(val_d_loss.detach().cpu().item())

        val_g_loss = compute_g_loss(
            G=generator,
            D=discriminator,
            bce_loss=bce_loss,
            noise=noise,
            crop_idx=crop_idx,
            real_labels=real_labels,
        )
        state.metrics.val_g_loss.append(val_g_loss.detach().cpu().item())

        val_e_loss = compute_e_loss(
            E=encoder,
            G=generator,
            l1_loss=l1_loss,
            real_images_crop=real_images_crop,
            lambda_1=state.lambda_1,
        )
        state.metrics.val_e_loss.append(val_e_loss.detach().cpu().item())

        val_sub_e_loss = compute_sub_e_loss(
            E=encoder,
            Sub_E=sub_encoder,
            G=generator,
            l1_loss=l1_loss,
            real_images=real_images,
            real_images_crop=real_images_crop,
            real_images_small=real_images_small,
            crop_idx=crop_idx,
            lambda_2=state.lambda_2,
        )
        state.metrics.val_sub_e_loss.append(val_sub_e_loss.detach().cpu().item())

    return state


def training_loop(
    models: DPModels,
    optimizers: DPOptimizers,
    dataloaders: DPDataLoaders,
    loss_fns: LossFNs,
    state: DPState,
    n_epochs: int,
    checkpoint_every_n_batch: Optional[int] = None,
    checkpoint_path: str = "dp_training/checkpoints",
) -> DPState:
    def c_name(x):
        return f"{checkpoint_path}/{x}_steps={state.step}.pth"

    for epoch in range(n_epochs):
        for i, batch in enumerate(dataloaders.train):
            state.step += 1
            state = training_step(
                models=models,
                optimizers=optimizers,
                dataloaders=dataloaders,
                loss_fns=loss_fns,
                state=state,
                batch=batch,
            )

            state.privacy_accountant.step(
                noise_multiplier=state.noise_multiplier, sample_rate=state.sample_rate
            )

            state = validation_step(
                models=models,
                dataloaders=dataloaders,
                loss_fns=loss_fns,
                state=state,
            )

            # if (
            #     checkpoint_every_n_batch is not None
            #     and state.step % checkpoint_every_n_batch == 0
            # ):
            # checkpoint_model(models.G, optimizers, state, c_name("generator"))
            # checkpoint_model(models.D, optimizers, state, c_name("discriminator"))
            # checkpoint_model(models.E, optimizers, state, c_name("encoder"))
            # checkpoint_model(models.Sub_E, optimizers, state, c_name("sub_encoder"))

    # checkpoint_model(models.G, optimizers, state, c_name("generator_final"))
    # checkpoint_model(models.D, optimizers, state, c_name("discriminator_final"))
    # checkpoint_model(models.E, optimizers, state, c_name("encoder_final"))
    # checkpoint_model(models.Sub_E, optimizers, state, c_name("sub_encoder_final"))

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
    checkpoint_at_epsilons: Optional[list[float]] = None,
) -> Tuple[DPState, float]:
    current_epsilon = state.privacy_accountant.get_epsilon(state.delta, alphas=alphas)

    dl_iter = iter(dataloaders.train)

    while current_epsilon < max_epsilon:
        batch = next(dl_iter)
        state = training_step(
            models=models,
            optimizers=optimizers,
            dataloaders=dataloaders,
            loss_fns=loss_fns,
            state=state,
            batch=batch,
        )

        state.privacy_accountant.step(
            noise_multiplier=state.noise_multiplier, sample_rate=state.sample_rate
        )

        current_epsilon = state.privacy_accountant.get_epsilon(
            state.delta, alphas=alphas
        )

    checkpoint_model(
        models.G,
        optimizers,
        state,
        f"{checkpoint_path}/generator_final_epsilon={current_epsilon}.pth",
    )

    return state, current_epsilon


def checkpoint_model(
    models: Union[nn.Module, GradSampleModule],
    optimizers: DPOptimizers,
    state: DPState,
    checkpoint_path: str,
):
    torch.save(
        {
            "state_dict": models.G.state_dict(),
            "optimizer": optimizers.g_opt.state_dict(),
            "epoch": state.epoch,
            "step": state.step,
            "metrics": state.metrics,
        },
        checkpoint_path,
    )
