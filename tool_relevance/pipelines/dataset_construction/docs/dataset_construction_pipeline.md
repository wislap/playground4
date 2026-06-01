# Dataset Generation Pipeline

This document defines the baseline pipeline for manufacturing a synthetic tool
relevance dataset. It complements `tool_metadata_design.md`: tool metadata stays
factual and local, while this pipeline uses LLM APIs only to create conversations
and relative judgments for training and evaluation data.

## Goals

- Build a fixed tool universe containing all agent tools and 100+ plugin tools.
- Generate conversations that have a reasonable chance of matching at least one
  real tool.
- Expose each sample to all agent tools plus a configurable random subset of
  plugin tools.
- Shuffle candidate tool order while preserving stable tool ids.
- Ask a judge LLM for relative raw scores over the visible candidate set.
- Calibrate raw scores into one continuous latent confidence value per
  `(sample, tool)` pair.
- Support parallel generation, judging, validation, calibration, and resume.

## Non-Goals

- Do not use LLM output to add or rewrite tool metadata.
- Do not create tool capabilities, side effects, risk levels, or categories.
- Do not train on judge explanations.
- Do not require all plugins to be visible in every sample.

## Artifacts

```text
data/
  tool_universes/
    neko_real_v1/
      manifest.json
      tools.jsonl
  tool_classifications/
    neko_real_v1.tool_taxonomy_v1.json
  conversations/
    generated.jsonl
  candidate_sets/
    sampled.jsonl
  judgments/
    raw_scores.jsonl
  calibrated/
    scores.jsonl
  datasets/
    train.jsonl
    valid.jsonl
    test.jsonl
  reports/
    generation_report.json
    calibration_report.json
    quality_report.json
```

All JSONL artifacts are append-friendly and keyed by stable ids so failed jobs
can be retried without rewriting completed work.

## Tool Universe

The tool universe is fixed before dataset generation starts. A universe is stored
as a directory: `manifest.json` describes the pool, and `tools.jsonl` stores one
tool per line.

```json
{
  "tool_universe_id": "neko_real_v1",
  "created_at": "2026-05-26T00:00:00Z",
  "kind": "real",
  "schema_version": "tool_universe_schema_v1",
  "source": {
    "type": "neko_repo"
  },
  "tool_count": 17
}
```

`tools.jsonl`:

```jsonl
{"tool_id":"agent.browser_use","kind":"agent","source_text":"kind: agent\nid: browser_use\n...","metadata":{"id":"browser_use"}}
{"tool_id":"plugin.web_search","kind":"plugin","source_text":"kind: plugin\nid: web_search\n...","metadata":{"id":"web_search"}}
```

Rules:

- `tool_id` is globally unique and stable.
- All agent tools are included.
- At least 100 plugin tools are included for the baseline.
- `source_text` is built from existing N.E.K.O facts only.
- LLM prompts may read `tool_id` and `source_text`, but may not rewrite them.
- Real and synthetic plugins share this same record shape. Their origin may be
  recorded in `manifest.json` or `metadata` for audit/splitting, but normal
  sampling and judging treat them as ordinary plugins.

### Synthetic Plugin Metadata

Synthetic plugins are not installed in N.E.K.O and do not have runtime state.
Their metadata must therefore be static and descriptive only.

Allowed synthetic plugin fields:

```json
{
  "id": "daily_briefing",
  "name": "每日简报",
  "type": "plugin",
  "description": "汇总用户一天开始前可能关心的信息，生成简短每日简报。",
  "short_description": "Generate a concise daily briefing from common user-facing information.",
  "keywords": ["每日简报", "daily briefing", "morning", "summary"]
}
```

Synthetic plugin `source_text`:

```text
kind: plugin
id: daily_briefing
name: 每日简报
type: plugin
description: 汇总用户一天开始前可能关心的信息，生成简短每日简报。
short_description: Generate a concise daily briefing from common user-facing information.
keywords: 每日简报, daily briefing, morning, summary
```

Synthetic plugins must not invent or include:

- `runtime.enabled`
- `runtime.auto_start`
- `status`
- real installation state
- real process state

For synthetic data, "entering the system" is represented only by candidate-set
sampling. It is not represented by runtime metadata.

## Tool Classification View

Before conversation generation starts, the pipeline builds a classification view
over the fixed tool universe. This is a dataset-generation control artifact, not
production metadata and not a scorer feature.

```json
{
  "tool_universe_id": "neko_real_v1",
  "taxonomy_version": "tool_taxonomy_v1",
  "categories": [
    {
      "tool_id": "agent.browser_use",
      "kind": "agent",
      "primary_category": "web_retrieval",
      "secondary_categories": [],
      "generation_tags": ["agent", "browser", "web"],
      "confidence": 0.95,
      "source": "heuristic"
    }
  ]
}
```

Rules:

- Every tool in the universe must have exactly one classification row.
- The classification may be produced by manual rules, heuristics, LLM review, or
  a mixed process.
- It must not create new tool capabilities.
- It may guide LLM-A target selection, scenario balancing, coverage reports, and
  plugin sampling weights.
- It must not be returned by the training dataloader as model input.

## Configuration

The pipeline should be controlled by one Hydra/YAML config.

```yaml
dataset:
  id: tool_relevance_synth_v1
  output_dir: data
  seed: 42
  train_ratio: 0.9
  valid_ratio: 0.05
  test_ratio: 0.05

tool_universe:
  id: neko_real_v1
  path: data/tool_universes/neko_real_v1

tool_classification:
  taxonomy_version: tool_taxonomy_v1
  path: data/tool_classifications/neko_real_v1.tool_taxonomy_v1.json

generation:
  num_conversations: 20000
  target_tool_mode_ratio: 0.85
  no_suitable_candidate_ratio: 0.12
  languages:
    zh-CN: 0.75
    en-US: 0.20
    mixed: 0.05
  max_messages: 6

candidate_sampling:
  plugin_sample_rate: 0.10
  target_force_rate: 0.80
  weight_decay_on_select: 0.70
  min_plugin_exposures: 50
  min_plugin_targeted: 20
  shuffle_candidates: true

llm:
  generator:
    provider: relay
    model: generator-model
    temperature: 0.8
    max_concurrency: 32
    timeout_seconds: 90
    retries: 3
  judge:
    provider: relay
    model: judge-model
    temperature: 0.0
    max_concurrency: 16
    timeout_seconds: 120
    retries: 3
    repeated_runs:
      train: 1
      valid: 2
      test: 3

calibration:
  method: local_z_global_rank_gaussian
  output_transform: latent
  clip_percentile: 0.001

quality:
  require_high_confidence_ratio: 0.85
  high_raw_score_threshold: 80
  max_duplicate_similarity: 0.92
  max_position_bias_delta: 20
```

## Stage 1: Conversation Generation

LLM-A sees the full tool universe. Its job is to create natural conversations
that have a reasonable positive target.

Input:

- All `tool_id` and `source_text` values.
- A sampled `target_tool_id` or small target tool set.
- Scenario recipe such as direct request, vague reference, multi-turn context,
  image-present context, cancelled/redacted turn, or no-tool-needed turn.

Output:

```json
{
  "conversation_id": "conv_000001",
  "messages": [
    {
      "role": "user",
      "text": "帮我打开网页查一下这个项目的最新 release"
    }
  ],
  "trigger": "turn_end",
  "generation_hint": {
    "target_tool_ids": ["agent.browser_use", "plugin.web_search"],
    "scenario_id": "web_lookup_direct",
    "language": "zh-CN"
  },
  "provenance": {
    "model": "generator-model",
    "prompt_version": "generator_v1",
    "seed": 1001
  }
}
```

`generation_hint` is for coverage accounting and candidate sampling only. It is
never shown to LLM-B and is never returned by the training dataloader.

## Stage 2: Candidate Sampling

Each sample exposes:

```text
all agent tools + sampled plugin tools
```

The plugin sample count is:

```text
k = ceil(num_plugins * plugin_sample_rate)
```

Plugin selection uses dynamic weighted sampling without replacement. Initial
weights are equal:

```text
w_i = 1.0
```

After a plugin is selected, remove part of its mass and redistribute that mass
to unselected plugins:

```python
selected = weighted_sample_without_replacement(plugin_ids, k, weights)

removed_mass = 0.0
for tool_id in selected:
    old = weights[tool_id]
    weights[tool_id] *= weight_decay_on_select
    removed_mass += old - weights[tool_id]

unselected = [p for p in plugin_ids if p not in selected]
for tool_id in unselected:
    weights[tool_id] += removed_mass / len(unselected)
```

To preserve positive coverage, target plugins from `generation_hint` may be
forced into the candidate set according to `target_force_rate`. Forced target
plugins still count toward `k`.

Candidate order is shuffled with a recorded seed:

```json
{
  "sample_id": "sample_000001",
  "conversation_id": "conv_000001",
  "candidate_set": {
    "sampling_seed": 38192,
    "plugin_sample_rate": 0.1,
    "tool_ids": [
      "plugin.pdf_reader",
      "agent.browser_use",
      "plugin.web_search",
      "agent.computer_use"
    ]
  }
}
```

## Stage 3: LLM Judge

LLM-B sees only the conversation and the sampled candidate tools. It does not
see `generation_hint`.

Input shape:

```text
Conversation:
...

Candidate tools, in the sampled shuffled order:
[1] tool_id: plugin.pdf_reader
source_text: ...

[2] tool_id: agent.browser_use
source_text: ...
```

LLM-B produces raw relative scores:

```json
{
  "sample_id": "sample_000001",
  "conversation_id": "conv_000001",
  "judge_run_id": "judge_000001_r1",
  "candidate_tool_order": [
    "plugin.pdf_reader",
    "agent.browser_use",
    "plugin.web_search",
    "agent.computer_use"
  ],
  "scores": [
    {
      "tool_id": "plugin.pdf_reader",
      "raw_score": 8
    },
    {
      "tool_id": "agent.browser_use",
      "raw_score": 92
    },
    {
      "tool_id": "plugin.web_search",
      "raw_score": 74
    },
    {
      "tool_id": "agent.computer_use",
      "raw_score": 31
    }
  ],
  "provenance": {
    "model": "judge-model",
    "prompt_version": "judge_v1"
  }
}
```

Raw scores are in `[0, 100]`. They are not training labels. They are only used
as relative evidence for calibration.

Judge prompt rules:

- Score every visible candidate tool.
- Prefer the best suitable tool with a high raw score, usually above 80.
- Use all-low scores only when no visible candidate can reasonably handle the
  request.
- Penalize passive or irrelevant tools.
- Do not infer capabilities absent from `source_text`.
- Keep the judgment relative to the visible candidate set.

## Stage 4: Score Aggregation

For repeated judge runs, aggregate raw scores per `(sample_id, tool_id)`:

```text
aggregated_raw_score = median(raw_score_run_1, raw_score_run_2, ...)
```

Record disagreement:

```text
judge_disagreement = max(raw_scores) - min(raw_scores)
```

Samples with high disagreement can be downweighted, sent for another judge run,
or excluded from valid/test splits.

## Stage 5: Gaussian Calibration

The final dataset stores one continuous latent `confidence` value per candidate
tool. It is not bounded to `[0, 1]`.

Calibration:

```text
aggregated_raw_score
  -> per-sample z-score over candidate_set
  -> global rank Gaussian transform
  -> confidence
```

Per-sample normalization:

```python
local_z_i = (raw_i - mean(raw_scores_for_sample)) / std(raw_scores_for_sample)
```

If all raw scores are identical, mark the sample invalid or set all `local_z_i`
to `0` and exclude it from calibration statistics.

Global rank Gaussian transform:

```python
p_i = (rank(local_z_i) - 0.5) / N
confidence_i = normal_inverse_cdf(p_i)
```

Optionally clip extreme percentiles:

```text
p_i in [clip_percentile, 1 - clip_percentile]
```

Runtime probability is derived later by one configured transform:

```python
probability = sigmoid(confidence)
```

or:

```python
probability = 0.5 * tanh(confidence) + 0.5
```

The project should pick one transform and keep it consistent across training
and inference.

## Final Dataset Format

```json
{
  "sample_id": "sample_000001",
  "split": "train",
  "tool_universe_id": "tool_universe_v1",
  "conversation": {
    "trigger": "turn_end",
    "messages": [
      {
        "role": "user",
        "text": "帮我打开网页查一下这个项目的最新 release"
      }
    ],
    "features": {
      "latest_user_has_image": false,
      "attachment_count": 0
    }
  },
  "candidate_set": {
    "plugin_sample_rate": 0.1,
    "sampling_seed": 38192,
    "tool_ids": [
      "plugin.pdf_reader",
      "agent.browser_use",
      "plugin.web_search",
      "agent.computer_use"
    ]
  },
  "targets": [
    {
      "tool_id": "plugin.pdf_reader",
      "confidence": -1.31
    },
    {
      "tool_id": "agent.browser_use",
      "confidence": 1.46
    },
    {
      "tool_id": "plugin.web_search",
      "confidence": 0.63
    },
    {
      "tool_id": "agent.computer_use",
      "confidence": -0.78
    }
  ],
  "quality": {
    "top_raw_score": 92,
    "judge_disagreement": 4,
    "has_high_confidence_candidate": true
  },
  "provenance": {
    "conversation_model": "generator-model",
    "judge_model": "judge-model",
    "generator_prompt_version": "generator_v1",
    "judge_prompt_version": "judge_v1",
    "calibration_version": "local_z_global_rank_gaussian_v1"
  }
}
```

## Dataloader Contract

The training dataloader resolves `candidate_set.tool_ids` against the fixed tool
universe and returns:

```python
{
    "sample_ids": list[str],
    "query_texts": list[str],
    "candidate_tool_ids": list[list[str]],
    "candidate_source_texts": list[list[str]],
    "confidence": FloatTensor,       # [batch, max_candidates]
    "candidate_mask": BoolTensor,    # [batch, max_candidates]
}
```

It must not return `generation_hint`, judge explanations, prompt text, or raw
LLM responses as model features.

## Parallel Execution

The pipeline is a DAG:

```text
tool_universe
  -> tool_classification
  -> conversations
  -> candidate_sets
  -> judgments
  -> calibration
  -> splits
  -> reports
```

Parallelizable stages:

- Conversation generation: independent by `conversation_id`.
- Candidate sampling: deterministic by `sample_id` seed.
- Judging: independent by `sample_id` and `judge_run_id`.
- Validation: independent per artifact row.
- Calibration: global reduce step after all valid judgments are available.

Recommended worker model:

- Use async LLM clients with bounded semaphores per provider/model.
- Keep one append-only JSONL writer task per artifact.
- Store idempotency keys for every LLM request.
- Retry transient failures with exponential backoff.
- Write failed jobs to `*.failed.jsonl`.

Each request should include:

```json
{
  "request_id": "judge.sample_000001.run_1",
  "idempotency_key": "dataset_v1:judge:sample_000001:run_1:prompt_hash"
}
```

Resume behavior:

- Load completed ids from existing JSONL artifacts.
- Skip completed ids with valid schema.
- Retry failed or invalid ids.
- Never regenerate downstream artifacts unless their upstream input hash changed.

## Quality Gates

Required checks:

- Every `tool_id` exists in the tool universe.
- Every sample includes all agent tools.
- Each sample includes approximately `plugin_sample_rate` of plugins.
- Candidate order is shuffled and recorded.
- LLM-B scores every candidate exactly once.
- Raw scores are numeric and in `[0, 100]`.
- At least the configured share of samples has a high raw-score candidate.
- Conversation text does not leak `target_tool_id`.
- Final confidence distribution is close to `N(0, 1)`.
- Per-plugin exposure counts meet minimum coverage.
- Per-plugin targeted counts meet minimum coverage.
- Near-duplicate conversations do not cross train/valid/test splits.
- Position bias is checked by rejudging a sample subset with different tool
  order.

Useful reports:

```text
global confidence mean/std/skew/kurtosis
per-tool exposure count
per-tool average confidence
per-scenario high-confidence rate
candidate target hit rate
judge disagreement distribution
position bias distribution
duplicate rate
LLM failure and retry rate
```

## Implementation Modules

Suggested package layout:

```text
src/tool_relevance_lab/dataset_generation/
  __init__.py
  config.py
  jsonl.py
  tool_universe.py
  conversation_generator.py
  candidate_sampler.py
  judge.py
  calibration.py
  quality.py
  split.py
  dataloader.py
  run.py
```

Suggested commands:

```bash
uv run python -m tool_relevance_lab.dataset_generation.run build-tool-universe
uv run python -m tool_relevance_lab.dataset_generation.run classify-tools
uv run python -m tool_relevance_lab.dataset_generation.run generate-synthetic-plugins
uv run python -m tool_relevance_lab.dataset_generation.run validate-synthetic-plugins
uv run python -m tool_relevance_lab.dataset_generation.run merge-tool-universes
uv run python -m tool_relevance_lab.dataset_generation.run generate-conversations
uv run python -m tool_relevance_lab.dataset_generation.run sample-candidates
uv run python -m tool_relevance_lab.dataset_generation.run judge
uv run python -m tool_relevance_lab.dataset_generation.run calibrate
uv run python -m tool_relevance_lab.dataset_generation.run build-dataset
uv run python -m tool_relevance_lab.dataset_generation.run report
```

The first implementation can make each command resumable and single-purpose.
After that, add an `all` command that runs the full DAG.
