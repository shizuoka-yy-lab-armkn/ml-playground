import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf
from torch import Tensor, nn

from ml_playground.models.mstcn import MultiStageModel
from ml_playground.proj.ymc_bfa_imu.config import ConfigSchema


class MstcnBinaryClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = MultiStageModel(
            num_stages=3,
            num_layers=9,
            in_feat_dim=18,
            num_f_maps=48,
            bottleneck_dim=32,
            out_feat_dim=64,
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        """(B, 18, T) -> (B, 2)"""
        out = self.encoder(x)
        out = self.fc(out)
        return out


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
