# V4.1 Hybrid Calibration and Two-MLP Report

Date: 2026-05-31

## Summary

The first V4 runs underperformed v2.2b mostly because the V4 final label was
calibrated with a per-sample relative ranking transform. That transform made
no-tool conversations dirty: when every candidate had a very low raw
`final_preference`, the least-bad candidate was still promoted to a high
calibrated confidence.

We fixed the V4 calibration without rerunning LLM-B:

```text
raw final_preference
-> absolute-strength mapping
-> optionally mix a small amount of relative rank only when group max raw is high
-> global rank-gaussian calibration
```

This keeps the global label distribution close to Gaussian while preserving the
meaning of "all tools are low" in no-tool / companionship-only scenes.

## Dataset Diagnosis

Old V4 calibration problem:

- Many no-tool samples had raw top scores in the `1..3` range.
- Per-sample ranking still made those low raw scores become high confidence.
- Example pattern: rationale said "do not recommend", but raw
  `final_preference = 1.0`; after per-sample calibration, this could become a
  strong positive target.

Observed old V4 statistics:

```text
old V4 no_tool group max mean: 1.906
old V4 no_tool groups with max < 1: 20 / 283
```

After hybrid calibration:

```text
new V4 confidence mean: 0.0000
new V4 confidence std: 0.9981
new V4 min/max: -3.0902 / 3.0902
new V4 no_tool group max mean: 0.753
new V4 no_tool groups with max < 1: 204 / 283
```

Files:

- active labels: `data/runs/neko_v4_multiaxis_500/calibrated.jsonl`
- old backup: `data/runs/neko_v4_multiaxis_500/calibrated.rank_gaussian_old.jsonl`

## Control Experiment

To isolate whether the previous V4 drop came from the model or from labels, we
reran the original v2.2b architecture on the fixed V4 dataset.

Configuration:

```text
sequence_late_interaction
MLP head
lexical gated_add
pairwise_weight = 0.2
data = V4 hybrid calibrated
5-fold CV
```

Result:

| Metric | Old V4 Calibration | Hybrid Calibration |
|---|---:|---:|
| MAE | 0.670 | 0.549 |
| Spearman | 0.486 | 0.727 |
| no_tool_fp_rate | 0.400 | 0.192 |
| top1_match | 0.550 | 0.510 |
| top3/topk_recall | 0.700 | 0.690 |
| NDCG@5 | 0.749 | 0.757 |
| top5_regret | 0.092 | 0.062 |

Interpretation:

- The old label transform was genuinely corrupting the continuous target.
- Hybrid calibration greatly improves continuous calibration, rank correlation,
  and no-tool safety.
- Top1/top3 do not recover because V4 is no longer a forced-pick dataset. Some
  conversations should have no strong tool recommendation.

## V4.1 Architecture

V4.1 uses two MLP-style heads:

```text
shared late-interaction features
-> factor MLP -> 6 axis predictions
-> final MLP over factor predictions + factor interactions + projected features
-> final_preference
```

Configuration:

```text
model.kind = factorized_late_interaction
factor_hidden_dim = 128
final_head = "mlp"
final_feature_dim = 32
use_factor_interactions = true
loss = pointwise + factor supervision + LambdaNDCG
data = V4 hybrid calibrated
```

Run:

```text
experiments/tool_rerank_baseline/runs/jina_sequence_v41_two_mlp_hybrid_cv_gpu
```

## Results

| Metric | v2.2b on V4 Hybrid | V4.1 Two-MLP |
|---|---:|---:|
| MAE | 0.549 | 0.540 |
| Spearman | 0.727 | 0.741 |
| NDCG@3 | 0.734 | 0.756 |
| NDCG@5 | 0.757 | 0.775 |
| top1_match | 0.510 | 0.550 |
| top3/topk_recall | 0.690 | 0.770 |
| top3_regret | 0.160 | 0.096 |
| top5_regret | 0.062 | 0.019 |
| high_conf_precision | 0.762 | 0.781 |
| no_tool_fp_rate | 0.192 | 0.229 |
| bad_top3_rate | 0.130 | 0.110 |

## Interpretation

V4.1 is the better mainline candidate after calibration cleanup.

Strengths:

- Better top3 recall.
- Better NDCG.
- Much lower regret.
- Slightly better Spearman and MAE.
- Better high-confidence precision.

Weakness:

- `no_tool_fp_rate` is slightly worse than the v2.2b-on-hybrid control.

This suggests that the two-MLP factorized model uses the multi-axis information
productively, but it is more willing to activate a candidate. The next version
should keep the V4.1 architecture and add a conservative safety term or
threshold-aware loss for no-tool cases.

## Next Step

Try a V4.1 safety variant:

```text
V4.1 two-MLP
+ no-tool false-positive penalty
+ bad exposure penalty
```

Primary target:

```text
keep top3/topk_recall and regret close to V4.1
reduce no_tool_fp_rate below 0.19
```

Top1/top3 should remain secondary metrics for V4. The primary V4 metrics should
be:

- Spearman;
- NDCG@5;
- top5 regret;
- no_tool_fp_rate;
- high-confidence precision;
- bad_top3_rate.
