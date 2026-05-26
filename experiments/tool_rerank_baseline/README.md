# Tool Rerank Baseline

Research-first baseline for N.E.K.O tool relevance training.

The first loop is intentionally small:

```text
data/runs/neko_v3_500
  -> grouped pair examples
  -> TF-IDF/SVD or Jina embeddings
  -> [conv, tool, abs diff, product]
  -> MLP confidence regressor
```

Run data inspection:

```bash
uv run python experiments/tool_rerank_baseline/inspect_data.py \
  --config experiments/tool_rerank_baseline/config.toml
```

Run training:

```bash
uv run python experiments/tool_rerank_baseline/train.py \
  --config experiments/tool_rerank_baseline/config.toml
```

Analyze a finished run:

```bash
uv run python experiments/tool_rerank_baseline/analyze_run.py \
  --run-dir experiments/tool_rerank_baseline/runs/tfidf_pair_mlp_v1
```

Use `encoder.backend = "jina"` when local model loading is ready. Keep `tfidf` as the
no-download fallback for quick iteration.
