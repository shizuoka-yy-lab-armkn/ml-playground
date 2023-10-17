import logging
import time
from pathlib import Path
from typing import Callable

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from ml_playground.dataset.hymenoptera_image_dataset import HymenopteraImageDataset
from ml_playground.repometa import DATA_ROOT
from ml_playground.types import PHASES
from ml_playground.util.logging import create_colored_handler
from ml_playground.util.random import fix_seed

SEED = 1933
_log = logging.getLogger(__name__)


def cmd_train(model_save_dir: str | Path, *, num_epochs: int, batch_size: int) -> None:
    logging.basicConfig(level=logging.INFO, handlers=[create_colored_handler()])
    fix_seed(SEED)

    data_dir = DATA_ROOT / "hymenoptera"
    assert data_dir.is_dir(), f"Directory '{data_dir}' does not exist"

    datasets = {phase: HymenopteraImageDataset(data_dir, phase) for phase in PHASES}
    _log.info(f"{datasets=}")

    dataloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        for phase in PHASES
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet18 = models.resnet18(weights="IMAGENET1K_V1")
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    resnet18 = resnet18.to(device)
    best_train_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        _log.info("--------- epoch %d/%d ----------", epoch + 1, num_epochs)

        start_at = time.time()
        train_acc, val_acc = train_1epoch(
            resnet18,
            criterion,
            optimizer,
            lr_scheduler,
            device,
            dataloaders["train"],
            dataloaders["val"],
        )
        elapsed = time.time() - start_at
        _log.info("epoch took time: %.2f sec", elapsed)

        best_train_acc = max(best_train_acc, train_acc)
        best_val_acc = max(best_val_acc, val_acc)
        _log.info(f"{best_train_acc=:.4f}, {best_val_acc=:.4f}")

        save_path = model_save_dir / f"epoch-{epoch + 1:03}.pt"
        torch.save(resnet18.state_dict(), save_path)


def train_1epoch(
    model: models.ResNet,
    criterion: Callable,
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[float, float]:
    """Returns (train_acc, val_acc)"""
    train_loss_sum = 0.0
    train_correct_count = 0
    train_data_count = 0
    model.train()

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs: torch.Tensor = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # collect train result
        _, pred_indexes = torch.max(outputs, 1)
        train_loss_sum += loss.item() * inputs.size(0)
        train_correct_count += torch.sum(pred_indexes == labels).item()
        train_data_count += len(labels)

    train_loss_mean = train_loss_sum / train_data_count
    train_acc = train_correct_count / train_data_count
    _log.info("(train) Loss=%.4f  Acc=%.4f", train_loss_mean, train_acc)

    val_loss_sum = 0.0
    val_correct_count = 0
    val_data_count = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, pred_indexes = torch.max(outputs, 1)
            val_loss_sum += loss.item() * inputs.size(0)
            val_correct_count += torch.sum(pred_indexes == labels).item()
            val_data_count += len(labels)

    val_loss_mean = val_loss_sum / val_data_count
    val_acc = val_correct_count / val_data_count
    _log.info("(val)  Loss=%.4f  Acc=%.4f", val_loss_mean, val_acc)

    return train_acc, val_acc


if __name__ == "__main__":
    fire.Fire({
        "train": cmd_train,
    })
