from pathlib import Path

from pydantic.dataclasses import dataclass

PROJECT_NAME = "ymc_bfa_imu"
PROJECT_DIR = Path(__file__).parent
CONFIG_DIR = PROJECT_DIR / "configs"


@dataclass
class TrainConfig:
    batch_size: int = 12
    max_epoch: int = 150
    min_epoch: int = 30
    lr: float = 3e-3
    weight_decay: float = 1e-5
    lr_decay_step_size: int = 20


@dataclass
class ConfigSchema:
    train: TrainConfig
    exp_name: str
    seed: int = 42
    early_stopping_patience: int = 5
