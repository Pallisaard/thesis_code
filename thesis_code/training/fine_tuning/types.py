from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lightning import LightningModule
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer
from opacus.data_loader import DPDataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from thesis_code.models.gans import LitHAGAN
from .utils import tree_key_map


### COMMON DATATYPES
@dataclass
class LossFNs:
    l1: nn.L1Loss = nn.L1Loss()
    bce: nn.BCELoss = nn.BCELoss()


@dataclass
class TrainMetrics:
    d_loss: list[float] = []
    g_loss: list[float] = []
    e_loss: list[float] = []
    sub_e_loss: list[float] = []
    total_loss: list[float] = []
    epsilon: list[float] = []

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "d_loss": self.d_loss,
            "g_loss": self.g_loss,
            "e_loss": self.e_loss,
            "sub_e_loss": self.sub_e_loss,
            "total_loss": self.total_loss,
            "epsilon": self.epsilon,
        }


@dataclass
class ValMetrics:
    d_loss: list[float] = []
    g_loss: list[float] = []
    e_loss: list[float] = []
    sub_e_loss: list[float] = []
    total_loss: list[float] = []
    d_accuracy: list[float] = []

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "val_d_loss": self.d_loss,
            "val_g_loss": self.g_loss,
            "val_e_loss": self.e_loss,
            "val_sub_e_loss": self.sub_e_loss,
            "d_accuracy": self.d_accuracy,
        }


@dataclass
class TrainingStats:
    log_dir: Path = Path("lightning/logs")
    val_every_n_steps: Optional[int] = None
    checkpoint_every_n_steps: Optional[int] = None
    current_epsilon: float = 0.0
    epoch: int = 0
    step: int = 0
    train_metrics: TrainMetrics = TrainMetrics()
    val_metrics: ValMetrics = ValMetrics()


### DP DATATYPES
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
class DPState:
    privacy_accountant: RDPAccountant
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    sample_rate: float
    alphas: list[float]
    lambdas: float
    device: str
    latent_dim: int

    training_stats: TrainingStats = TrainingStats()


@dataclass
class DPModels:
    G: nn.Module
    D: GradSampleModule
    E: GradSampleModule
    Sub_E: GradSampleModule

    def to_lit(self, state: DPState) -> LightningModule:
        def fix_state_dict(state_dict):
            return tree_key_map(lambda k: k.replace("_module.", ""), state_dict)

        model = LitHAGAN(
            latent_dim=state.latent_dim, lambda_1=state.lambdas, lambda_2=state.lambdas
        )
        model.G.load_state_dict(fix_state_dict(self.G.state_dict()))
        model.D.load_state_dict(fix_state_dict(self.D.state_dict()))
        model.E.load_state_dict(fix_state_dict(self.E.state_dict()))
        model.Sub_E.load_state_dict(fix_state_dict(self.Sub_E.state_dict()))

        return model


### NO DP DATATYPES
@dataclass
class NoDPModels:
    G: nn.Module
    D: nn.Module
    E: nn.Module
    Sub_E: nn.Module

    def to_lit(self, state: DPState) -> LightningModule:
        def fix_state_dict(state_dict):
            return tree_key_map(lambda k: k.replace("_module.", ""), state_dict)

        model = LitHAGAN(
            latent_dim=state.latent_dim, lambda_1=state.lambdas, lambda_2=state.lambdas
        )
        model.G.load_state_dict(fix_state_dict(self.G.state_dict()))
        model.D.load_state_dict(fix_state_dict(self.D.state_dict()))
        model.E.load_state_dict(fix_state_dict(self.E.state_dict()))
        model.Sub_E.load_state_dict(fix_state_dict(self.Sub_E.state_dict()))

        return model


@dataclass
class NoDPOptimizers:
    g_opt: optim.Optimizer
    d_opt: optim.Optimizer
    e_opt: optim.Optimizer
    sub_e_opt: optim.Optimizer


@dataclass
class NoDPDataLoaders:
    train: DataLoader
    val: DataLoader


# @dataclass
# class NoDPState:
#     lambdas: float
#     device: str
#     latent_dim: int

#     training_stats: TrainingStats = TrainingStats()
