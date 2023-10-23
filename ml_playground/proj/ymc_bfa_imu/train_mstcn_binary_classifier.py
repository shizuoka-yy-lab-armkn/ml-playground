import os

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from torch import Tensor, nn

from ml_playground.proj.ymc_bfa_imu.config import CONFIG_DIR, PROJECT_NAME, ConfigSchema
from ml_playground.proj.ymc_bfa_imu.data import ImuDataModule
from ml_playground.proj.ymc_bfa_imu.models import MstcnBinaryClassifier
from ml_playground.util.pl_callback import create_common_callbacks


class LitImuBinaryClassifier(pl.LightningModule):

    def __init__(self, cfg: ConfigSchema) -> None:
        super().__init__()
        # convert to dict using OmegaConf
        self.save_hyperparameters(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        self.cfg = cfg
        self.model = MstcnBinaryClassifier()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _forward_and_calc_metrics(
        self, inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns (loss, accuracy)"""
        logits = self.model(inputs)
        loss: Tensor = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=-1)
        acc = torch.sum(preds == targets) / targets.size(0)
        return loss, acc

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
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
            optimizer, step_size=self.cfg.train.lr_decay_step_size, gamma=0.8
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}  # type: ignore


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
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
        deterministic=False,
        callbacks=create_common_callbacks(
            proj=PROJECT_NAME,
            exp=cfg.exp_name,
            early_stopping_patience=cfg.early_stopping_patience,
        ),
        max_epochs=cfg.train.max_epoch,
        min_epochs=cfg.train.min_epoch,
        logger=[wandb_logger, comet_logger],
    )
    dm = ImuDataModule(bsz=cfg.train.batch_size)
    model = LitImuBinaryClassifier(cfg)

    wandb_logger.watch(model.model)

    print("###################### Starting trainer.fit() ########################")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    ConfigStore.instance().store(name="config_schema", node=ConfigSchema)
    main()
