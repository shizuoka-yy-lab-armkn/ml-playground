import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

from ml_playground.repometa import REPO_ROOT


def create_common_callbacks(
    *, proj: str, exp: str, early_stopping_patience: int
) -> list[pl.Callback]:
    return [
        EarlyStopping("val_loss", mode="min", patience=early_stopping_patience),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=REPO_ROOT / "checkpoints" / proj / exp,
            filename="{epoch:03}-{step:04}-{val_acc:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
    ]
