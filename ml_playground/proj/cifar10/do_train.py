import os
from pathlib import Path

import comet_ml as _
import hydra
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.transforms.v2 as transformsv2
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from omegaconf import OmegaConf
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from ml_playground.proj.cifar10.config import (
    CONFIG_DIR,
    PROJECT_NAME,
    AugmentMethod,
    ConfigSchema,
    ModelType,
)
from ml_playground.repometa import DATA_ROOT, REPO_ROOT

NUM_CLASSES = 10


def _create_untrained_model(model: ModelType) -> nn.Module:
    if model == ModelType.ResNet18:
        # see: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html#Resnet
        net = torchvision.models.resnet18(num_classes=NUM_CLASSES)
        net.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        net.maxpool = nn.Identity()  # type: ignore
        return net


def _get_train_transform(aug: AugmentMethod) -> transformsv2.Transform:
    if aug == AugmentMethod.RandAugment:
        return transformsv2.Compose([
            transformsv2.RandAugment(),
            transformsv2.ToTensor(),
            cifar10_normalization(),
        ])


def _get_val_transform() -> transformsv2.Transform:
    return transformsv2.Compose([
        transformsv2.ToTensor(),
        cifar10_normalization(),
    ])


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, data_root: Path, bsz: int, aug: AugmentMethod) -> None:
        super().__init__()
        self.data_root = data_root
        self.bsz = bsz
        self.aug = aug

    def setup(self, stage: str) -> None:
        del stage
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_root),
            train=True,
            download=True,
            transform=_get_train_transform(self.aug),
        )
        self.val_dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_root),
            train=False,
            download=True,
            transform=_get_val_transform(),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.bsz,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.bsz,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


class LitCIFAR10Classifier(pl.LightningModule):

    def __init__(self, cfg: ConfigSchema) -> None:
        super().__init__()

        # convert to dict using OmegaConf
        self.save_hyperparameters(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        self.cfg = cfg
        self.model = _create_untrained_model(cfg.model)

    def forward(self, x: Tensor) -> Tensor:
        """
        x:   [bsz, channel=3, H=32, W=32]
        out: [bsz, num_classes]
        """
        out = self.model(x)
        return F.log_softmax(out, dim=-1)

    def _forward_and_calc_metrics(
        self, inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns (loss, accuracy)"""
        logits = self.model(inputs)
        loss: Tensor = F.nll_loss(logits, targets)

        preds = torch.argmax(logits, dim=-1)
        acc = torchmetrics.functional.accuracy(
            preds, targets, "multiclass", num_classes=NUM_CLASSES
        )

        return loss, acc

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        if self.current_epoch == 0 and batch_idx == 0:
            self.print("---------- train_step() ------------")
            self.print(f"{self.current_epoch=}, {batch_idx=}")
            self.print(f"{batch[0].size()=}, {batch[1].size()=}")

        inputs, targets = batch
        loss, acc = self._forward_and_calc_metrics(inputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        if self.current_epoch == 0 and batch_idx == 0:
            self.print("---------- validation_step() ------------")
            self.print(f"{self.current_epoch=}, {batch_idx=}")
            self.print(f"{batch[0].size()=}, {batch[1].size()=}")

        inputs, targets = batch
        loss, acc = self._forward_and_calc_metrics(inputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # cfg = self.cfg.train
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=cfg.lr,
        #     weight_decay=self.cfg.train.weight_decay,
        # )
        #
        # # doc: https://timm.fast.ai/SGDR#CosineLRScheduler
        # scheduler = CosineLRScheduler(
        #     optimizer,
        #     t_initial=cfg.max_epoch,
        #     lr_min=cfg.lr_min,
        #     warmup_t=cfg.lr_warmup_epochs,
        #     warmup_lr_init=cfg.lr_start,  # type: ignore
        #     warmup_prefix=True,
        # )
        #
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,  # type: ignore
        #         "interval": "epoch",
        #         "strict": True,
        #     },
        # }
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.train.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.cfg.train.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.cfg.train.max_epoch,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict} #type: ignore


    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric) -> None:
        del metric
        scheduler.step(self.current_epoch)


def _create_callbacks() -> list[pl.Callback]:
    return [
        EarlyStopping("val_loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=REPO_ROOT / "checkpoints" / PROJECT_NAME,
            filename="{epoch:03}-{step:04}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=True,
            save_last=True,
        ),
    ]


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: ConfigSchema) -> None:
    print(type(cfg), cfg)
    OmegaConf.to_object(cfg)  # validate with pydantic

    pl.seed_everything(seed=cfg.seed)

    if cfg.just_print_model:
        model = _create_untrained_model(cfg.model)
        print(model)
        return

    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=cfg.exp_name,
        group=cfg.model.name,
    )
    comet_logger = CometLogger(
        project_name=PROJECT_NAME,
        experiment_name=cfg.exp_name,
        api_key=os.environ["COMET_API_KEY"],
        save_dir="comet_logs",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        deterministic=True,
        callbacks=_create_callbacks(),
        max_epochs=cfg.train.max_epoch,
        logger=[wandb_logger, comet_logger],
    )
    dm = CIFAR10DataModule(
        DATA_ROOT / "cifar10", bsz=cfg.train.batch_size, aug=cfg.train.aug_method
    )
    dm.prepare_data()
    model = LitCIFAR10Classifier(cfg)

    wandb_logger.watch(model.model)

    print("###################### Starting trainer.fit() ########################")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    ConfigStore.instance().store(name="config_schema", node=ConfigSchema)
    main()
