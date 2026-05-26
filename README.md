# Tool Relevance Lab

Small PyTorch workspace for experimenting with context-aware tool relevance
scoring before integrating anything into N.E.K.O.

## Setup

```bash
uv sync
```

## Sanity Check

```bash
uv run python -m tool_relevance_lab.torch_sanity
uv run pytest
```

## Dataset Generation

Design docs:

- `docs/tool_metadata_design.md`
- `docs/dataset_generation_pipeline.md`

Local utility commands currently implemented:

```bash
uv run python -m tool_relevance_lab.dataset_generation.run import-neko-tools \
  --neko-root /home/yun_wan/python_programe/N.E.K.O \
  --output data/tool_universes/neko_real_v1 \
  --tool-universe-id neko_real_v1

uv run python -m tool_relevance_lab.dataset_generation.run classify-tools \
  --tool-universe data/tool_universes/neko_real_v1 \
  --output data/tool_classifications/neko_real_v1.tool_taxonomy_v1.json

uv run python -m tool_relevance_lab.dataset_generation.run inspect-tool-universes \
  --tool-universe data/tool_universes/neko_real_v1 \
  --tool-universe data/tool_universes/synthetic_plugins_v1 \
  --tool-universe-id runtime_tool_pool_v1

uv run python -m tool_relevance_lab.dataset_generation.run calibrate \
  --judgments data/judgments/raw_scores.jsonl \
  --output data/calibrated/scores.jsonl

uv run python -m tool_relevance_lab.dataset_generation.run report \
  --tool-universe data/tool_universes/neko_real_v1 \
  --candidate-sets data/candidate_sets/sampled.jsonl \
  --judgments data/judgments/raw_scores.jsonl \
  --output data/reports/quality_report.json
```

Generate synthetic plugins with an OpenAI-compatible relay:

```bash
TOOL_GEN_API_KEY=... TOOL_GEN_BASE_URL=https://relay.example/v1 \
uv run python -m tool_relevance_lab.dataset_generation.run generate-synthetic-plugins \
  --real-tool-universe data/tool_universes/neko_real_v1 \
  --taxonomy data/tool_classifications/neko_real_v1.tool_taxonomy_v1.json \
  --output data/tool_universes/synthetic_plugins_v1 \
  --tool-universe-id synthetic_plugins_v1 \
  --model your-generator-model \
  --count 40 \
  --batch-size 10
```

Or copy and fill the TOML config:

```bash
cp conf/synthetic_plugin_generation.example.toml conf/synthetic_plugin_generation.toml
uv run python scripts/generate_synthetic_plugins.py --config conf/synthetic_plugin_generation.toml
```

Validate generated tools:

```bash
uv run python -m tool_relevance_lab.dataset_generation.run validate-synthetic-plugins \
  --synthetic-tool-universe data/tool_universes/synthetic_plugins_v1 \
  --real-tool-universe data/tool_universes/neko_real_v1 \
  --output data/reports/synthetic_plugins_v1.validation.json
```

Commands that consume a tool universe can load multiple sources directly:

```bash
uv run python -m tool_relevance_lab.dataset_generation.run classify-tools \
  --tool-universe data/tool_universes/neko_real_v1 \
  --tool-universe data/tool_universes/synthetic_plugins_v1 \
  --tool-universe-id runtime_tool_pool_v1 \
  --output data/tool_classifications/runtime_tool_pool_v1.tool_taxonomy_v1.json
```
