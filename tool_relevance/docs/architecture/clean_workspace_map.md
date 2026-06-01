# Clean Workspace Map

## Canonical Modules

### V2.2b Relevance Scorer

Location:

```text
tool_relevance/modules/v22b_relevance_scorer/
```

Purpose:

- Load the V3 dataset.
- Render conversations and tools into pair examples.
- Build lexical and policy features.
- Train the Jina sequence late-interaction scorer with grouped cross-validation.
- Report ranking, calibration, activation, and bad-exposure metrics.

Active config:

```text
tool_relevance/modules/v22b_relevance_scorer/configs/active/
```

### V5 Action Policy

Location:

```text
tool_relevance/modules/v5_action_policy/
```

Purpose:

- Consume V2.2b relevance scores and current plugin enabled states.
- Recommend plugin enable or disable actions.
- Generate judged action slates.
- Train the action policy with multi-scale GRPO feedback.

## Canonical Dataset

### V3 Neko 500

Location:

```text
tool_relevance/datasets/v3_neko_500/
```

Contents:

- `run/`: generated conversations, candidate sets, judgments, calibration, manifest.
- `tool_universes/`: real and synthetic tool universe inputs.
- `tool_classifications/`: taxonomy views for generation and reporting.
- `reports/`: fixed inspection and similarity reports.

## Dataset Construction Pipeline

Location:

```text
tool_relevance/pipelines/dataset_construction/
```

Purpose:

- Import or merge tool universes.
- Generate synthetic plugins.
- Generate conversations.
- Sample candidate tools.
- Judge candidate sets.
- Calibrate labels.
- Produce quality reports.

Pipeline configs live in `configs/` and must be examples or environment-driven.

## Analysis Tools

Location:

```text
tool_relevance/tools/analysis/
```

Purpose:

- Analyze training runs.
- Plot metrics.
- Inspect bad cases.
- Recompute metrics.
- Compute slice metrics.

## Legacy Boundary

The original project directories remain in place for compatibility and audit history.
They are not the target structure for new work.

New work should be added under `tool_relevance/`.
