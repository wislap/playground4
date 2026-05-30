# Tool Rerank Baseline

Research-first baseline for N.E.K.O tool relevance training.

Current training policy: all model training must go through grouped 5-fold
cross-validation. Single split training, repeated seed runners, parallel ad-hoc
runners, ridge sweeps, and non-CV training configs have been removed to keep
validation strict and comparable.

The active loop is:

```text
data/runs/neko_v3_500
  -> grouped conversation-level pair examples
  -> fixed holdout groups excluded from CV
  -> 5-fold CV over the CV group pool
  -> shared frozen Jina sequence encoding for the CV training pool
  -> fold-local lexical features fit only on that fold's training groups
  -> sequence late-interaction regressor with gated lexical fusion
  -> per-fold checkpoint evaluation and aggregate CV report
```

Inspect data:

```bash
uv run python experiments/tool_rerank_baseline/inspect_data.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22b_cv.toml
```

Run 5-fold CV on CPU:

```bash
uv run python experiments/tool_rerank_baseline/cross_validate.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22b_cv.toml
```

Run 5-fold CV on GPU:

```bash
uv run python experiments/tool_rerank_baseline/cross_validate.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22b_cv_gpu.toml
```

Analyze a finished fold or run directory:

```bash
uv run python experiments/tool_rerank_baseline/analyze_run.py \
  --run-dir experiments/tool_rerank_baseline/runs/jina_sequence_lexical_gated_v22b_cv/fold_01
```

`training_core.py` is internal library code used by `cross_validate.py`; it is
not a training entrypoint.
