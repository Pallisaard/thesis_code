from typing import Any, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from opacus import GradSampleModule

from .types import (
    DPState,
    NoDPState,
)


def checkpoint_dp_model(
    models: Union[nn.Module, GradSampleModule],
    state: DPState,
    checkpoint_path: str,
):
    torch.save(
        {
            "state_dict": models.G.state_dict(),
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


def checkpoint_no_dp_model(
    models: nn.Module,
    state: NoDPState,
    checkpoint_path: str,
):
    torch.save(
        {
            "state_dict": models.G.state_dict(),
            "epoch": state.training_stats.epoch,
            "step": state.training_stats.step,
        },
        checkpoint_path,
    )

    train_metrics_file = f"{checkpoint_path}_train_metrics.csv"
    save_dict_as_csv(state.training_stats.train_metrics.to_dict(), train_metrics_file)
    val_metrics_file = f"{checkpoint_path}_val_metrics.csv"
    save_dict_as_csv(state.training_stats.val_metrics.to_dict(), val_metrics_file)


def load_checkpoint(
    models: Union[nn.Module, GradSampleModule],
    state: DPState,
    checkpoint_path: str,
) -> Tuple[Union[nn.Module, GradSampleModule], DPState, float]:
    checkpoint = torch.load(checkpoint_path)
    models.G.load_state_dict(checkpoint["state_dict"])
    state.training_stats.epoch = checkpoint["epoch"]
    state.training_stats.step = checkpoint["step"]
    return models, state, checkpoint["epsilon"]


def save_dict_as_csv(data_dict: dict[str, Any], csv_file_path: str):
    """
    Saves a dictionary as a CSV file. Keys are column names and values are column values.
    """
    df = pd.DataFrame(data_dict)
    df.to_csv(csv_file_path, index=False)
