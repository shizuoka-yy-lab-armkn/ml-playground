import enum
from pathlib import Path

from pydantic.dataclasses import dataclass

PROJECT_NAME = "cifar10"
PROJECT_DIR = Path(__file__).parent
CONFIG_DIR = PROJECT_DIR / "configs"


# 本当は typing.Literal を使いたいけど hydra が対応していないので仕方なく enum を使っている


class AugmentMethod(enum.Enum):
    RandAugment = enum.auto()


class ModelType(enum.Enum):
    ResNet18 = enum.auto()


@dataclass
class TrainConfig:
    batch_size: int
    aug_method: AugmentMethod
    max_epoch: int
    lr_warmup_epochs: int
    lr: float
    lr_min: float
    lr_start: float
    weight_decay: float


@dataclass
class ConfigSchema:
    train: TrainConfig
    model: ModelType
    exp_name: str
    seed: int = 42
    just_print_model: bool = False
