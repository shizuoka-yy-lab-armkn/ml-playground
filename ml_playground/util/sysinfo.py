from logging import getLogger

import torch

_log = getLogger(__name__)


def log_cuda_info() -> None:
    _log.info(f"{torch.cuda.is_available()=}")
    _log.info(f"{torch.backends.cudnn.is_available()=}")
    _log.info(f"{torch.backends.cudnn.version()=}")
