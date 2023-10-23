from typing import Literal, NewType

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ml_playground.repometa import DATA_ROOT

Actor = Literal["Anakamura", "ksuzuki", "master", "ueyama"]
DateStr = str
ActionID = NewType("ActionID", int)

RECORDS: list[tuple[Actor, DateStr, list[int]]] = [
    ("Anakamura", "0628", [1, 2, 3, 4, 6, 7, 8, 9, 10]),
    ("Anakamura", "0703", [1, 2, 3, 5, 6, 7, 8, 9, 11]),
    ("Anakamura", "0628", [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]),
    ("ksuzuki", "0627", [1, 2, 4, 5, 6, 7, 8, 9, 10]),
    ("ksuzuki", "0704", [1, 2, 3, 4, 5, 7, 9, 10, 11, 13]),
    ("ksuzuki", "0706", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ("master", "0620", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("ueyama", "0628", [2, 5, 6, 8, 9, 10, 11, 13]),
    ("ueyama", "0705", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("ueyama", "0706", [1, 2, 3, 4, 5, 7, 8, 9, 10]),
]


# 初心者->0, 熟練者->1
Label = Literal[0, 1]


class ImuDataSet1(Dataset):
    TRAIN: list[tuple[Actor, DateStr, ActionID, Label]] = [
        # master
        ("master", "0620", ActionID(1), 1),
        ("master", "0620", ActionID(3), 1),
        ("master", "0620", ActionID(5), 1),
        ("master", "0620", ActionID(7), 1),
        ("master", "0620", ActionID(9), 1),
        ("ueyama", "0706", ActionID(1), 1),
        ("ueyama", "0706", ActionID(3), 1),
        ("ueyama", "0706", ActionID(5), 1),
        ("ueyama", "0706", ActionID(7), 1),
        # beginner
        ("Anakamura", "0628", ActionID(1), 0),
        ("Anakamura", "0628", ActionID(3), 0),
        ("Anakamura", "0628", ActionID(6), 0),
        ("Anakamura", "0628", ActionID(8), 0),
        ("ksuzuki", "0627", ActionID(1), 0),
        ("ksuzuki", "0627", ActionID(4), 0),
        ("ksuzuki", "0627", ActionID(6), 0),
        ("ksuzuki", "0627", ActionID(8), 0),
    ]

    VAL: list[tuple[Actor, DateStr, ActionID, Label]] = [
        # master
        ("master", "0620", ActionID(2), 1),
        ("master", "0620", ActionID(4), 1),
        ("master", "0620", ActionID(6), 1),
        ("master", "0620", ActionID(8), 1),
        ("master", "0620", ActionID(10), 1),
        ("ueyama", "0706", ActionID(2), 1),
        ("ueyama", "0706", ActionID(4), 1),
        ("ueyama", "0706", ActionID(8), 1),
        ("ueyama", "0706", ActionID(10), 1),
        # beginner
        ("Anakamura", "0628", ActionID(2), 0),
        ("Anakamura", "0628", ActionID(4), 0),
        ("Anakamura", "0628", ActionID(7), 0),
        ("Anakamura", "0628", ActionID(9), 0),
        ("ksuzuki", "0627", ActionID(2), 0),
        ("ksuzuki", "0627", ActionID(5), 0),
        ("ksuzuki", "0627", ActionID(7), 0),
        ("ksuzuki", "0627", ActionID(9), 0),
    ]

    def __init__(self, *, train: bool) -> None:
        super().__init__()

        src = self.TRAIN if train else self.VAL

        self.dataset: list[tuple[Tensor, Label]] = []

        for actor, date, aid, label in src:
            path = DATA_ROOT / "ymc-bfa-imu" / "12" / actor / f"{date}_{aid:03}.npy"
            dat = Tensor(np.load(path))
            assert dat.size(0) == 18
            self.dataset.append((dat, label))

    def __getitem__(self, index):
        assert type(index) is int
        assert 0 <= index < len(self.dataset)
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


class ImuDataModule(pl.LightningDataModule):

    def __init__(self, bsz: int) -> None:
        super().__init__()
        self.bsz = bsz
        self.train_dataset = ImuDataSet1(train=True)
        self.val_dataset = ImuDataSet1(train=False)

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Label]]) -> tuple[Tensor, Tensor]:
        """
        pad and create mask
        """
        max_len = max(x.size(-1) for x, _ in batch)
        xs = torch.stack([F.pad(x, pad=(0, max_len - x.size(-1))) for x, _ in batch])
        ys = torch.LongTensor([y for _, y in batch])
        assert xs.size() == (len(batch), 18, max_len)
        return xs, ys

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.bsz,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.bsz,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
