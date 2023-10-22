import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ml_playground.dataset.hymenoptera_image_dataset import HymenopteraImageDataset
from ml_playground.repometa import DATA_ROOT


class HymenopteraDataModule(pl.LightningDataModule):
    data_dir = DATA_ROOT / "hymenoptera"

    def __init__(self, bsz: int) -> None:
        super().__init__()
        self.bsz = bsz
        self.train_dataset = HymenopteraImageDataset(self.data_dir, "train")
        self.val_dataset = HymenopteraImageDataset(self.data_dir, "val")

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
