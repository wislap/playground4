from __future__ import annotations

from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ExperimentConfigSummary:
    seed: int
    experiment_name: str
    batch_size: int


def summarize_config(cfg: DictConfig) -> ExperimentConfigSummary:
    return ExperimentConfigSummary(
        seed=int(cfg.seed),
        experiment_name=str(cfg.experiment.name),
        batch_size=int(cfg.data.batch_size),
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    summary = summarize_config(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(
        "summary="
        f"seed:{summary.seed} "
        f"experiment:{summary.experiment_name} "
        f"batch_size:{summary.batch_size}"
    )


if __name__ == "__main__":
    main()
