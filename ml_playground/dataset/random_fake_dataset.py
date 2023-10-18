import numpy as np
import torch.utils.data
from torch.utils.data.dataset import Dataset

_T = tuple[int, float]


class RandomFakeDataset(Dataset[_T]):

    def __init__(self) -> None:
        pass

    def __getitem__(self, index: int) -> _T:
        """Returns (index, random_value)"""

        # ruff: noqa: T201
        # getLogger() を使った場合、DataLoader で複数workerを使うとログが表示されない
        print("RandomDataset.__getitem__:", torch.utils.data.get_worker_info())
        return index, np.random.uniform(-1, 1)

    def __len__(self) -> int:
        return 12
