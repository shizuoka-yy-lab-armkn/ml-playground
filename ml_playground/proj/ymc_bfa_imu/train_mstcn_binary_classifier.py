import os

import hydra
import lightning.pytorch as pl
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import CometLogger, WandbLogger
from omegaconf import OmegaConf

from ml_playground.proj.ymc_bfa_imu.config import CONFIG_DIR, PROJECT_NAME, ConfigSchema
from ml_playground.proj.ymc_bfa_imu.data import ImuDataModule
from ml_playground.proj.ymc_bfa_imu.models import LitImuBinaryClassifier
from ml_playground.util.pl_callback import create_common_callbacks


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
