from typing import Any, Tuple, Union
from collections.abc import Callable

from jax import tree_map
import pandas as pd
import torch
import torch.nn as nn
from opacus import GradSampleModule
from lightning import LightningModule

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
):
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


def save_dict_as_csv(data_dict: dict[str, Any], csv_file_path: str):
    """
    Saves a dictionary as a CSV file. Keys are column names and values are column values.
    """
    df = pd.DataFrame(data_dict)
    df.to_csv(csv_file_path, index=False)
