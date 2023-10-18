from pathlib import Path
from typing import Callable

from PIL.Image import Image
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ml_playground.types import PhaseType

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _get_transform(phase: PhaseType) -> Callable[[Image], Tensor]:
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD, inplace=True),
        ])
    elif phase == "val":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD, inplace=True),
        ])
    else:
        raise TypeError(f"unexpected phase '{phase}'")


class HymenopteraImageDataset(ImageFolder):

    def __init__(self, root: Path, phase: PhaseType) -> None:
        super().__init__(str(root / phase), transform=_get_transform(phase))
