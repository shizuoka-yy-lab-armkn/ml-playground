import enum
from pathlib import Path

from pydantic.dataclasses import dataclass

PROJECT_NAME = "cifar10"
PROJECT_DIR = Path(__file__).parent
CONFIG_DIR = PROJECT_DIR / "configs"


class AugmentMethod(enum.Enum):
    TimmRandAugment = enum.auto()


@dataclass
class TrainConfig:
    batch_size: int
    aug_method: AugmentMethod


@dataclass
class ConfigSchema:
    train: TrainConfig
