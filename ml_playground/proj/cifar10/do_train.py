import os

import hydra
import lightning.pytorch as pl
import torch
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
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn

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
    transformsv2.RandAugment()
    if aug == AugmentMethod.RandAugment:
        return transformsv2.Compose([
            transformsv2.RandAugment(),
            transformsv2.ToTensor(),
            cifar10_normalization(),
        ])

    raise ValueError(f"Invalid augmentation method: got={aug}")


def _get_val_transform() -> transformsv2.Transform:
    return transformsv2.Compose([
        transformsv2.ToTensor(),
        cifar10_normalization(),
    ])


class LitCIFAR10Classifier(pl.LightningModule):

    def __init__(self, cfg: ConfigSchema) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg

        self.model = _create_untrained_model(cfg.model)

        self.mixup = Mixup(
            mixup_alpha=0.0,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.0,
            mode="batch",
            label_smoothing=0,
            num_classes=NUM_CLASSES,
        )

        self.criterion = SoftTargetCrossEntropy()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _forward_and_calc_metrics(
        self, inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns (loss, accuracy)"""
        logits = self.model(inputs)
        loss: Tensor = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=-1)
        acc = torchmetrics.functional.accuracy(preds, targets, "multiclass")

        return loss, acc

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        if self.current_epoch <= 1 and batch_idx <= 1:
            self.print("---------- train_step() ------------")
            self.print(f"{self.current_epoch=}, {batch_idx=}")
            self.print(f"{batch[0].size()=}, {batch[1].size()=}")

        inputs, targets = batch
        inputs, targets = self.mixup(inputs, targets)
        loss, acc = self._forward_and_calc_metrics(inputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        if self.current_epoch <= 1 and batch_idx <= 1:
            self.print("---------- validation_step() ------------")
            self.print(f"{self.current_epoch=}, {batch_idx=}")
            self.print(f"{batch[0].size()=}, {batch[1].size()=}")

        inputs, targets = batch
        loss, acc = self._forward_and_calc_metrics(inputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        cfg = self.cfg.train
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.lr,
            weight_decay=self.cfg.train.weight_decay,
        )

        # doc: https://timm.fast.ai/SGDR#CosineLRScheduler
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.max_epoch,
            lr_min=cfg.lr_min,
            warmup_t=cfg.lr_warmup_epochs,
            warmup_lr_init=cfg.lr_start,  # type: ignore
            warmup_prefix=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,  # type: ignore
                "interval": "epoch",
                "strict": True,
            },
        }

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
        data_dir=str(DATA_ROOT / "cifar10"),
        batch_size=cfg.train.batch_size,
        num_workers=2,
        train_transforms=_get_train_transform(cfg.train.aug_method),
        val_transforms=_get_val_transform(),
    )
    model = LitCIFAR10Classifier(cfg)

    wandb_logger.watch(model.model)
    trainer.fit(model, dm)


if __name__ == "__main__":
    ConfigStore.instance().store(name="config_schema", node=ConfigSchema)
    main()
