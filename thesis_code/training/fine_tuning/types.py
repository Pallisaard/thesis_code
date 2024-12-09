from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer
from opacus.data_loader import DPDataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


### COMMON DATATYPES
@dataclass
class LossFNs:
    l1: nn.L1Loss = nn.L1Loss()
    bce: nn.BCELoss = nn.BCELoss()


@dataclass
class TrainMetrics:
    d_loss: list[float] = field(default_factory=list)
    g_loss: list[float] = field(default_factory=list)
    e_loss: list[float] = field(default_factory=list)
    sub_e_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)

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
    d_loss: list[float] = field(default_factory=list)
    g_loss: list[float] = field(default_factory=list)
    e_loss: list[float] = field(default_factory=list)
    sub_e_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    d_accuracy: list[float] = field(default_factory=list)

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
    train_metrics: TrainMetrics = field(default_factory=TrainMetrics)
    val_metrics: ValMetrics = field(default_factory=ValMetrics)


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

    training_stats: TrainingStats = field(default_factory=TrainingStats)


@dataclass
class DPModels:
    G: nn.Module
    D: GradSampleModule
    E: GradSampleModule
    Sub_E: GradSampleModule


### NO DP DATATYPES
@dataclass
class NoDPModels:
    G: nn.Module
    D: nn.Module
    E: nn.Module
    Sub_E: nn.Module


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
