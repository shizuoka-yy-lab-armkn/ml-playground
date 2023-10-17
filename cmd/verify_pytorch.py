# ruff: noqa: T201 (allow print)
import logging

import torch

from ml_playground.util.logging import create_colored_handler
from ml_playground.util.sysinfo import log_cuda_info

logging.basicConfig(level=logging.DEBUG, handlers=[create_colored_handler()])

_log = logging.getLogger(__name__)


def main() -> None:
    log_cuda_info()

    print("------------")
    _log.debug("debug log message test")
    _log.info("info log message test")
    _log.warning("warning log message test")
    _log.error("error log message test")
    _log.critical("critical log message test")

    print("------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(2, 5).to(device)
    b = torch.randn(5, 3).to(device)
    dot = a @ b
    print(f"{device=}")
    print(f"{a=}")
    print(f"{b=}")
    print(f"{dot=}")


if __name__ == "__main__":
    main()
    pass
