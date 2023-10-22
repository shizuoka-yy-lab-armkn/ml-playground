from pathlib import Path

from pydantic.dataclasses import dataclass

PROJECT_NAME = "hymenoptera"
PROJECT_DIR = Path(__file__).parent
CONFIG_DIR = PROJECT_DIR / "configs"


@dataclass
class TrainConfig:
    batch_size: int = 128
    max_epoch: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_decay_step_size: int = 7


@dataclass
class ConfigSchema:
    train: TrainConfig
    exp_name: str
    seed: int = 42
