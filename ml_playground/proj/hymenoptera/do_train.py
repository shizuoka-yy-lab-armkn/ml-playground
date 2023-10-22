import os

import hydra
import lightning.pytorch as pl
import torch
import torchvision
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from torch import Tensor, nn

from ml_playground.proj.cifar10.config import PROJECT_NAME
from ml_playground.proj.hymenoptera.config import CONFIG_DIR, ConfigSchema
from ml_playground.proj.hymenoptera.data import HymenopteraDataModule
from ml_playground.repometa import REPO_ROOT


class LitHymenopteraClassifier(pl.LightningModule):

    def __init__(self, cfg: ConfigSchema) -> None:
        super().__init__()
        # convert to dict using OmegaConf
        self.save_hyperparameters(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        self.cfg = cfg
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _create_model() -> nn.Module:
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _forward_and_calc_metrics(
        self, inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns (loss, accuracy)"""
        logits = self.model(inputs)
        loss: Tensor = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=-1)
        acc = torch.sum(preds == targets)
        return loss, acc

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        if self.current_epoch == 0 and batch_idx == 0:
            self.print("---------- train_step() ------------")
            self.print(f"{self.current_epoch=}, {batch_idx=}")
            self.print(f"{batch[0].size()=}, {batch[1].size()=}")

        inputs, targets = batch
        loss, acc = self._forward_and_calc_metrics(inputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        if self.current_epoch == 0 and batch_idx == 0:
            self.print("---------- validation_step() ------------")
            self.print(f"{self.current_epoch=}, {batch_idx=}")
            self.print(f"{batch[0].size()=}, {batch[1].size()=}")

        inputs, targets = batch
        loss, acc = self._forward_and_calc_metrics(inputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.train.lr,
            momentum=0.9,
            weight_decay=self.cfg.train.weight_decay,
        )
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.train.lr_decay_step_size, gamma=0.1
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}  # type: ignore


def _create_callbacks() -> list[pl.Callback]:
    return [
        EarlyStopping("val_loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=REPO_ROOT / "checkpoints" / PROJECT_NAME,
            filename="{epoch:03}-{step:04}-{val_acc:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=True,
            save_last=True,
        ),
    ]


@hydra.main(config_path=str(CONFIG_DIR), version_base=None)
def main(cfg: ConfigSchema) -> None:
    print(type(cfg), cfg)
    OmegaConf.to_object(cfg)  # validate with pydantic

    pl.seed_everything(seed=cfg.seed)

    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=cfg.exp_name,
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
    dm = HymenopteraDataModule(bsz=cfg.train.batch_size)
    model = LitHymenopteraClassifier(cfg)

    wandb_logger.watch(model.model)

    print("###################### Starting trainer.fit() ########################")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    ConfigStore.instance().store(name="config_schema", node=ConfigSchema)
    main()
