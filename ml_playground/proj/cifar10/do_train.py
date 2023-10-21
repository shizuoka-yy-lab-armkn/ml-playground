import hydra
import torch
import torchvision
import torchvision.transforms.v2 as transformsv2
from hydra.core.config_store import ConfigStore
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader

from ml_playground.proj.cifar10.config import CONFIG_DIR, AugmentMethod, ConfigSchema
from ml_playground.repometa import DATA_ROOT


def _get_augment_method(aug: AugmentMethod) -> transformsv2.Transform:
    if aug == AugmentMethod.TimmRandAugment:
        return transformsv2.Compose([
            rand_augment_transform(
                config_str="rand-m9-mstd0.5",
                hparams={"translate_const": 117, "img_mean": (124, 116, 104)},
            ),
            transformsv2.ToTensor(),
        ])

    raise ValueError(f"Invalid augmentation method: got={aug}")


def _create_data_loader_pair(
    batch_size: int, aug: AugmentMethod
) -> tuple[DataLoader, DataLoader]:
    """Creates a pair of `(train_loader, val_loader)`"""
    transform = _get_augment_method(aug)

    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_ROOT),
        download=True,
        train=True,
        transform=transform,
    )
    val_set = torchvision.datasets.CIFAR10(
        root=str(DATA_ROOT),
        download=True,
        train=False,
        transform=transformsv2.ToTensor(),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: ConfigSchema) -> None:
    print(type(cfg), cfg.train)
    # train_loader, val_loader = _create_data_loader_pair(
    #     batch_size=cfg.train.batch_size, aug=cfg.train.aug_method
    # )


if __name__ == "__main__":
    ConfigStore.instance().store(name="config_schema", node=ConfigSchema)
    main()
