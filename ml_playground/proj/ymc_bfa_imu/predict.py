import hydra
import lightning.pytorch as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from ml_playground.proj.ymc_bfa_imu.config import CONFIG_DIR, ConfigSchema
from ml_playground.proj.ymc_bfa_imu.data import ImuDataSet1
from ml_playground.proj.ymc_bfa_imu.models import LitImuBinaryClassifier
from ml_playground.repometa import REPO_ROOT


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: ConfigSchema) -> None:
    print(type(cfg), cfg)
    OmegaConf.to_object(cfg)  # validate with pydantic

    pl.seed_everything(seed=cfg.seed)

    dataset = ImuDataSet1(full=True)
    model = LitImuBinaryClassifier.load_from_checkpoint(
        REPO_ROOT
        / "checkpoints"
        / "ymc_bfa_imu"
        / "mstcn-relu-000"
        / "epoch=078-step=0158-val_acc=0.88.ckpt",
        cfg=cfg,
    )
    model.freeze()

    for i in range(len(dataset)):
        inputs, targets, name = dataset[i]
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to("cuda")
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1)
        print(name, preds.item(), sep=",")


if __name__ == "__main__":
    ConfigStore.instance().store(name="config_schema", node=ConfigSchema)
    main()
