from omegaconf import OmegaConf

from tool_relevance_lab.config_sanity import summarize_config


def test_hydra_style_config_summary() -> None:
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "experiment": {"name": "smoke"},
            "data": {"batch_size": 4},
        }
    )

    summary = summarize_config(cfg)

    assert summary.seed == 7
    assert summary.experiment_name == "smoke"
    assert summary.batch_size == 4
