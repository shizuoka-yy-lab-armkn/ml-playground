# ruff: noqa: T201
import logging

import fire
import torch.utils.data
from torch.utils.data import DataLoader

from ml_playground.dataset.random_fake_dataset import RandomFakeDataset
from ml_playground.util.logging import create_colored_handler
from ml_playground.util.random import fix_seed

_log = logging.getLogger(__name__)


def main(num_workers: int = 4, seed: int = 1933) -> None:
    logging.basicConfig(level=logging.DEBUG, handlers=[create_colored_handler()])
    fix_seed(seed)

    _log.info(f"{seed=}")
    _log.info(f"{num_workers=}")

    dataset = RandomFakeDataset()
    _log.info(f"{dataset[0]=}")
    _log.info(f"{dataset[1]=}")

    print("------------- DataLoader without worker_init_fn -------------")
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=num_workers,
        batch_size=1,
    )

    # メインスレッド(プロセス) では get_worker_info() は None になるはず
    _log.info(f"{torch.utils.data.get_worker_info()=}")

    for i, batch in enumerate(dataloader):
        print(i, batch)

    _log.info(f"{torch.utils.data.get_worker_info()=}")


if __name__ == "__main__":
    fire.Fire(main)
