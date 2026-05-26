# Tool Rerank Baseline

Research-first baseline for N.E.K.O tool relevance training.

The first loop is intentionally small:

```text
data/runs/neko_v3_500
  -> grouped pair examples
  -> TF-IDF/SVD pair MLP, or Jina token sequence late interaction
  -> optional lexical side-channel: word/char ngram cosine + BM25 + token overlap
  -> confidence regressor
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

Run the cached Jina sequence baseline:

```bash
uv run python experiments/tool_rerank_baseline/train.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_v2.toml
```

Run Jina sequence plus lexical side-channel:

```bash
uv run python experiments/tool_rerank_baseline/train.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_lexical_v21.toml
```

Run gated lexical fusion with lexical feature dropout:

```bash
uv run python experiments/tool_rerank_baseline/train.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22.toml
```

Run gated lexical concat, which keeps lexical evidence explicit but scales it
with a learned gate:

```bash
uv run python experiments/tool_rerank_baseline/train.py \
  --config experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_concat_v23.toml
```

Analyze a finished run:

```bash
uv run python experiments/tool_rerank_baseline/analyze_run.py \
  --run-dir experiments/tool_rerank_baseline/runs/tfidf_pair_mlp_v1
```

Use `encoder.backend = "jina"` or `"jina_sequence"` when local model loading is ready.
Keep `tfidf` as the no-download fallback for quick iteration.
