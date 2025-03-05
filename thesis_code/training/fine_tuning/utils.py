from pathlib import Path
from typing import Any, Tuple, Union
from collections.abc import Callable

from jax import tree_map
import pandas as pd
import torch
from lightning import LightningModule

from thesis_code.dataloading.utils import save_mri
from thesis_code.models.gans import LitHAGAN
from thesis_code.models.gans.hagan.backbone.Model_HA_GAN_256 import (
    Generator,
    Discriminator,
    Encoder,
    Sub_Encoder,
)
from thesis_code.models.gans.hagan.dp_safe_backbone.Model_HA_GAN_256 import (
    Generator as safe_Generator,
    Discriminator as safe_Discriminator,
    Encoder as safe_Encoder,
    Sub_Encoder as safe_Sub_Encoder,
)

from .types import (
    DPModels,
    DPState,
    NoDPModels,
)


def tree_key_map(func: Callable[[str], Any], tree: Any) -> Any:
    if isinstance(tree, dict):
        return {tree_key_map(func, k): v for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(func(v) for v in tree)
    else:
        return func(tree)


tree_map = tree_map


def checkpoint_dp_model(
    models: Union[DPModels, NoDPModels],
    state: DPState,
    checkpoint_path: str,
    save_mri_example: bool = False,
):
    if save_mri_example:
        mri_path = Path(checkpoint_path).parent / "mri_example.nii.gz"
        save_mri(
            models.G.sample(2),
            mri_path,
        )

    torch.save(
        {
            "g_state_dict": models.G.state_dict(),
            "d_state_dict": models.D.state_dict(),
            "e_state_dict": models.E.state_dict(),
            "sub_e_state_dict": models.Sub_E.state_dict(),
            "epoch": state.training_stats.epoch,
            "step": state.training_stats.step,
            "epsilon": state.privacy_accountant.get_epsilon(state.delta),
        },
        checkpoint_path,
    )

    train_metrics_file = f"{checkpoint_path}_train_metrics.csv"
    save_dict_as_csv(state.training_stats.train_metrics.to_dict(), train_metrics_file)
    val_metrics_file = f"{checkpoint_path}_val_metrics.csv"
    save_dict_as_csv(state.training_stats.val_metrics.to_dict(), val_metrics_file)


def load_checkpoint(
    models: Union[DPModels, NoDPModels],
    state: DPState,
    checkpoint_path: str,
) -> Tuple[Union[DPModels, NoDPModels], DPState, float]:
    checkpoint = torch.load(checkpoint_path)
    models.G.load_state_dict(checkpoint["state_dict"])
    models.D.load_state_dict(checkpoint["state_dict"])
    models.E.load_state_dict(checkpoint["state_dict"])
    models.Sub_E.load_state_dict(checkpoint["state_dict"])
    state.training_stats.epoch = checkpoint["epoch"]
    state.training_stats.step = checkpoint["step"]
    return models, state, checkpoint["epsilon"]


def load_checkpoint_components(
    checkpoint_path: str,
    map_location: str = "auto",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    generator_dict = checkpoint["g_state_dict"]
    discriminator_dict = checkpoint["d_state_dict"]
    discriminator_dict = tree_key_map(lambda x: x.replace("_module.", ""), discriminator_dict)
    encoder_dict = checkpoint["e_state_dict"]
    encoder_dict = tree_key_map(lambda x: x.replace("_module.", ""), encoder_dict)
    sub_encoder_dict = checkpoint["sub_e_state_dict"]
    sub_encoder_dict = tree_key_map(lambda x: x.replace("_module.", ""), sub_encoder_dict)

    return generator_dict, discriminator_dict, encoder_dict, sub_encoder_dict


def save_dict_as_csv(data_dict: dict[str, Any], csv_file_path: str):
    """
    Saves a dictionary as a CSV file. Keys are column names and values are column values.
    """
    df = pd.DataFrame(data_dict)
    df.to_csv(csv_file_path, index=False)


def convert_models_to_lit(models: DPModels | NoDPModels, state: DPState) -> LightningModule:
    def fix_state_dict(state_dict):
        return tree_key_map(lambda k: k.replace("_module.", ""), state_dict)

    model = LitHAGAN(latent_dim=state.latent_dim, lambda_1=state.lambdas, lambda_2=state.lambdas)
    model.G.load_state_dict(fix_state_dict(models.G.state_dict()))
    model.D.load_state_dict(fix_state_dict(models.D.state_dict()))
    model.E.load_state_dict(fix_state_dict(models.E.state_dict()))
    model.Sub_E.load_state_dict(fix_state_dict(models.Sub_E.state_dict()))

    return model
