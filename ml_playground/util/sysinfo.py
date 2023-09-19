import torch


def print_cuda_info() -> None:
    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.backends.cudnn.m.version()=}")  # type:ignore
    print(f"{torch.backends.cudnn.m.is_available()=}")  # type:ignore
