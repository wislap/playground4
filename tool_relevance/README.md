# Tool Relevance

This directory is the clean, canonical workspace layout for the tool relevance project.

The old top-level directories are treated as legacy source material. New work should be
placed here using the domain-oriented structure below.

## Layout

```text
tool_relevance/
  modules/
    v22b_relevance_scorer/     Canonical V2.2b semantic relevance scorer.
    v5_action_policy/          V5 enable/disable action policy built on V2.2b scores.
  datasets/
    v3_neko_500/               Canonical V3 dataset and its tool universe inputs.
  pipelines/
    dataset_construction/      Dataset generation, candidate sampling, judging, calibration.
  tools/
    analysis/                  Offline analysis and report utilities.
  tests/                       Tests mirrored by domain.
  runtime/                     Local cache, run outputs, checkpoints, and generated artifacts.
  docs/
    architecture/              Architecture notes for this clean layout.
```

## Active Components

- `modules/v22b_relevance_scorer`: train and evaluate the active relevance model.
- `modules/v5_action_policy`: learn plugin state actions from V2.2b scores.
- `datasets/v3_neko_500`: the fixed V3 dataset used by V2.2b.
- `pipelines/dataset_construction`: build future datasets in a controlled way.
- `tools/analysis`: inspect trained runs and failure cases.

## Naming Rules

- Module source files carry the module prefix, for example `v22b_model.py`.
- Dataset folders include both version and identity, for example `v3_neko_500`.
- Legacy or non-active material must include `legacy` in the filename or folder name.
- Runtime artifacts belong under `runtime/`, not beside source files.
- Config files must never contain real API keys. Use environment variables.
