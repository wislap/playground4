# V3 Neko 500 Dataset

This is the canonical V3 dataset for the V2.2b relevance scorer.

## Contents

```text
run/
  conversations.jsonl
  candidate_sets.jsonl
  judgments.jsonl
  calibrated.jsonl
  manifest.json
  quality_report.json

tool_universes/
  neko_real_v1/
  synthetic_plugins_v1/
  synthetic_plugins_v2/
  synthetic_plugins_v3/
  synthetic_plugins_v4/

tool_classifications/
  *.tool_taxonomy_v1.json

reports/
  runtime_tool_pool_v4_inspection.json
  runtime_tool_pool_v4_similarity.json
```

## Contract

- This dataset is fixed input data, not scratch output.
- New generated datasets should go under `tool_relevance/runtime/outputs/` first.
- A generated dataset becomes canonical only after review and promotion here.
