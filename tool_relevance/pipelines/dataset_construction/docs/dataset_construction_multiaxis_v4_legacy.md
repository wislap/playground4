# Dataset Generation V4 Multi-Axis Labels

V4 upgrades the judge stage from one mixed relevance score to factorized,
continuous scores. The goal is to stop forcing LLM-B to compress capability,
intent, consent, cost, and N.E.K.O companionship style into one number too
early.

## Label Axes

Each candidate tool receives seven raw scores in `[0, 100]`:

- `capability_match`: whether the tool can perform the task if the task is
  actually desired.
- `action_demand`: whether the current conversation asks N.E.K.O to act.
- `target_specificity`: whether the request points to this tool's capability
  category.
- `consent_boundary`: whether this intervention is allowed by the context.
- `intervention_cost`: disruption, privacy, external side effects, or
  irreversibility. High means costly, not necessarily wrong.
- `companionship_fit`: whether using or mentioning the tool fits N.E.K.O's
  emotional companionship moment.
- `final_preference`: final recommendation strength after combining factors.

These are intentionally not treated as discrete labels. Every axis is
calibrated to a continuous rank-Gaussian latent value.

## Output Format

`calibrated.jsonl` remains backward compatible with the v2.2b training loader:

```json
{
  "sample_id": "sample_000001",
  "tool_id": "plugin.example",
  "confidence": 1.23,
  "raw_score": 91.0,
  "axis_scores": {
    "capability_match": 1.50,
    "action_demand": 0.84,
    "target_specificity": 1.31,
    "consent_boundary": 0.71,
    "intervention_cost": -0.12,
    "companionship_fit": 0.66,
    "final_preference": 1.23
  },
  "raw_scores": {},
  "local_z": {},
  "judge_disagreement": {}
}
```

`confidence` is an alias for `axis_scores.final_preference`, so existing
single-target experiments can continue unchanged. New experiments can read
`axis_scores` as high-order supervision or as an intermediate factor vector.

## Running

Use the v4 config template:

```bash
python -m tool_relevance_lab.dataset_generation.run run-pipeline \
  --config conf/dataset_pipeline_v4_multiaxis.example.toml
```

For a no-API smoke test:

```bash
python -m tool_relevance_lab.dataset_generation.run run-pipeline \
  --config conf/dataset_pipeline_v4_multiaxis.example.toml \
  --dry-run
```

The pipeline is still resumable. Existing completed rows in conversations,
candidate sets, and judgments are skipped by `sample_id`.

## Modeling Use

The intended next step is not to replace `final_preference` with hard rules.
Instead:

- keep `final_preference` as the primary continuous target;
- optionally train auxiliary heads for the six factor axes;
- learn a lightweight linear or gated interaction head from factor predictions
  to final preference;
- compare against the v2.2b single-label baseline under the existing 5-fold CV.
