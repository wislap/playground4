from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchEnvironment:
    version: str
    cuda_available: bool
    device: str


def torch_environment() -> TorchEnvironment:
    cuda_available = torch.cuda.is_available()
    return TorchEnvironment(
        version=torch.__version__,
        cuda_available=cuda_available,
        device="cuda" if cuda_available else "cpu",
    )


def main() -> None:
    env = torch_environment()
    x = torch.tensor([1.0, 2.0, 3.0], device=env.device)
    y = x.square().sum()
    print(f"torch={env.version} device={env.device} cuda={env.cuda_available} result={y.item():.1f}")


if __name__ == "__main__":
    main()
