# V2.2b Relevance Scorer

Canonical module for the V2.2b semantic relevance model.

## Files

- `source/v22b_dataset_loader.py`: loads V3 run artifacts and builds pair examples.
- `source/v22b_model.py`: encoders, lexical features, and scorer networks.
- `source/v22b_policy_features.py`: hand-built policy and consent features.
- `source/v22b_training_core.py`: training loops, encoding, checkpoint evaluation.
- `source/v22b_cross_validate.py`: grouped cross-validation entrypoint.
- `source/v22b_metrics.py`: ranking, calibration, activation, and exposure metrics.
- `source/v22b_inspect_dataset.py`: dataset inspection utility.

## Active Config

```text
configs/active/v22b_jina_sequence_lexical_policy_cv_lambda_gpu.toml
```

## Dataset

The active dataset is:

```text
tool_relevance/datasets/v3_neko_500/run/
```
