import torch

from ml_playground.util.sysinfo import print_cuda_info


def main() -> None:
    print_cuda_info()
    print("------------")
    x = torch.randn(2, 3)
    print(x)


if __name__ == "__main__":
    main()
    pass
