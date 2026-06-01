# Tool Rerank Baseline V2.2b Policy Feature Ablation Report

Date: 2026-05-30

This report records a controlled ablation on top of the V2.2b LambdaNDCG
baseline. The purpose is to test whether lightweight hand-designed policy
features help N.E.K.O's tool reranking task, and to identify which feature
groups are useful before moving toward a larger V3 feature design.

This is not a V3 architecture experiment. The base architecture, optimizer,
loss, dataset split, encoder, lexical features, and 5-fold CV protocol are kept
fixed. Only the added policy feature group changes.

## 1. Motivation

N.E.K.O is primarily an emotional companion with optional agent/tool execution.
This means tool relevance is not only semantic similarity:

- A user may mention a domain without authorizing a tool call.
- Emotional support or passive listening should often avoid tool activation.
- Some tools are risky because they imply external side effects, monitoring,
  sending messages, uploading data, or broader agentic control.
- The ranking target is continuous confidence, not a discrete class.

The previous V2.2b baseline already uses:

- frozen Jina sequence embeddings
- late interaction between conversation and tool sequences
- lexical BM25/ngram features through gated fusion
- SmoothL1 pointwise calibration plus LambdaNDCG@5 ranking loss

The question tested here is narrower:

```text
Can small policy features improve ranking or calibration when added to V2.2b,
without changing the model architecture?
```

## 2. Controlled Setup

Base config:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22b_cv_lambda_gpu.toml
```

Policy-feature configs:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v22b_cv_lambda_gpu.toml
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_conv_v22b_cv_lambda_gpu.toml
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_tool_v22b_cv_lambda_gpu.toml
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_interaction_v22b_cv_lambda_gpu.toml
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_cross_v22b_cv_lambda_gpu.toml
```

Common settings:

```text
dataset:        data/runs/neko_v3_500
tools:          149 unique tools
validation:     grouped 5-fold CV
holdout:        20% conversation groups excluded from CV
encoder:        jinaai/jina-embeddings-v5-text-nano-retrieval
max_length:     128
model:          sequence late-interaction regressor
head:           MLP
hidden_dim:     256
dropout:        0.1
lexical:        enabled, gated_add, dropout 0.05
optimizer:      AdamW
lr:             0.001
weight_decay:   0.0001
epochs:         8
loss:           0.7 SmoothL1 + 0.3 LambdaNDCG@5
```

Validation independence:

- Splits are grouped by `conversation_id`.
- Pairs from the same conversation do not cross fold boundaries.
- The held-out group pool is not used in CV.
- Jina sequence encoding is shared only for efficiency; fold-local lexical
  features are still fit only on each fold's training groups.

## 3. Feature Groups

Implementation:

```text
experiments/tool_rerank_baseline/policy_features.py
```

The policy features are metadata-light. They use only the existing conversation
text, scenario/relevance strings, tool id, and flattened tool text. No new tool
metadata is required.

The available groups are:

```text
conv          18 features
tool          20 features
interaction   11 features
cross          8 features
```

### conv

Conversation-side intent and boundary features:

```text
agentic task, companion context, boundary/refusal,
proactive/open-thread context, actionable/authorized request,
no-tool/no-reminder/no-upload/no-monitoring/no-search/no-file/no-planning/no-control,
just venting, quiet presence
```

These features do not inspect which tool is being scored. They are mostly
conversation-level priors.

### tool

Tool-side static domain and risk features:

```text
agent/plugin kind, file, browser, search, message, calendar, reminder,
monitoring, emotion, audio, camera, home, travel, code, research, study,
social, external, proactive
```

These features do not inspect the current conversation except through the final
model's learned interaction with other inputs. They are static tool priors.

### interaction

Explicit conversation-tool compatibility and conflict features:

```text
refusal-domain conflict, no-reminder vs reminder tool,
no-upload vs external tool, no-monitoring vs monitoring/emotion tool,
no-search vs search tool, no-file vs file tool,
no-planning vs travel/calendar tool, no-control vs external tool,
companion vs proactive tool, agentic-control opportunity,
recommendation opportunity
```

These are direct relative features for the pair being ranked.

### cross

Small multiplicative pair features:

```text
agentic x agent, companion x intrusive, boundary x external,
actionable x browser/file/agent, refusal x monitoring/message
```

These are narrower than `interaction` and mostly encode coarse pair products.

## 4. Results

Aggregate 5-fold CV mean:

```text
run           top1   top3   ndcg5   regret5  bad3  bad5  precision  hv@3   hv@5   mae    spearman
base_lambda   .800   .900   .8860   .0332    .02   .04   .718       .548   .713   .566   .697
policy_all    .750   .850   .8888   .0528    .02   .05   .748       .575   .735   .547   .707
conv-only     .770   .870   .8936   .0526    .02   .03   .779       .568   .754   .567   .704
tool-only     .770   .830   .8823   .0687    .02   .03   .854       .542   .727   .574   .688
interaction   .830   .890   .8868   .0332    .03   .05   .748       .553   .708   .561   .696
cross-only    .800   .910   .8790   .0514    .03   .05   .776       .564   .739   .585   .684
```

Delta against the V2.2b Lambda baseline:

```text
run           top1    top3    ndcg5    regret5  bad3    bad5    precision  hv@3    hv@5    mae      spearman
policy_all   -.050   -.050   +.0028   +.0196   +.000   +.010   +.0295     +.0266  +.0222  -.0183   +.0105
conv-only    -.030   -.030   +.0076   +.0195   +.000   -.010   +.0604     +.0198  +.0412  +.0016   +.0070
tool-only    -.030   -.070   -.0038   +.0355   +.000   -.010   +.1353     -.0059  +.0136  +.0086   -.0084
interaction  +.030   -.010   +.0008   +.0000   +.010   +.010   +.0295     +.0052  -.0051  -.0043   -.0013
cross-only   +.000   +.010   -.0070   +.0183   +.010   +.010   +.0578     +.0164  +.0259  +.0192   -.0133
```

Metric notes:

- `top1`: whether the predicted top tool is the highest-labeled tool in the
  candidate set.
- `top3`: whether the highest-labeled tool appears in predicted top 3.
- `ndcg5`: ranking quality within top 5.
- `regret5`: label gap between the true best tool and the best label inside
  predicted top 5. Lower is better.
- `bad3` / `bad5`: bad-tool exposure rate in top 3 / top 5. Lower is better.
- `precision`: high-confidence precision at the configured threshold.
- `hv@3` / `hv@5`: high-value recall at top 3 / top 5.
- `mae`: continuous confidence error. Lower is better.
- `spearman`: global rank correlation.

## 5. Interpretation

### 5.1 Full policy features are not a clean win

`policy_all` improves:

```text
ndcg5, high-confidence precision, high-value recall, MAE, Spearman
```

but damages:

```text
top1, top3, regret5, bad5
```

This means the full policy feature set improves broad calibration and global
ranking correlation, but hurts precise candidate-set ordering. For the current
reranking goal, this is not acceptable as a direct replacement for the V2.2b
baseline.

The likely reason is that the side channel is too unconstrained. Once all policy
features are concatenated into the same gated lexical path, the model can
overuse static or coarse policy signals even when the sequence encoder already
captures the local semantic match.

### 5.2 `tool-only` is overconfident static prior

`tool-only` produces the highest high-confidence precision:

```text
.718 -> .854
```

but it also causes the worst top3 and regret:

```text
top3:    .900 -> .830
regret5: .033 -> .069
```

This is a classic sign of an over-strong static tool prior. The model becomes
more confident about tools with attractive domain/risk signatures, but that
does not mean it ranks the correct tool for the current user turn.

For N.E.K.O this is especially dangerous: static plugin type is not enough to
decide whether a user wants action, quiet companionship, clarification, or no
tool at all.

Conclusion:

```text
Do not directly include tool-only static policy features in the main ranker,
unless they are heavily regularized, dropped out, or restricted to a separate
calibration/risk head.
```

### 5.3 `conv-only` is useful for calibration and coverage, not precise ranking

`conv-only` improves:

```text
ndcg5:     .886 -> .894
precision: .718 -> .779
hv@5:      .713 -> .754
bad5:      .040 -> .030
spearman:  .697 -> .704
```

but hurts:

```text
top1:    .800 -> .770
top3:    .900 -> .870
regret5: .033 -> .053
```

This suggests conversation-level policy priors are useful for understanding
whether the turn is generally tool-like, companion-like, or boundary-heavy.
However, because they are identical for every candidate tool in the same
conversation, they are weak for fine-grained tool ordering.

Conclusion:

```text
Conv features should be used as a weak calibration or gating signal, not as a
strong direct ranking feature.
```

### 5.4 `interaction-only` is the best direct ranking feature group

`interaction-only` is the only feature group that improves top1:

```text
top1: .800 -> .830
```

while preserving the most important regret metric:

```text
regret5: .033 -> .033
```

It only slightly lowers top3:

```text
top3: .900 -> .890
```

and keeps ndcg5 approximately unchanged:

```text
ndcg5: .8860 -> .8868
```

This is the cleanest signal in the ablation. It means relative pair features
that explicitly compare user boundaries/actionability against tool affordances
are useful. They encode information the sequence model may not reliably infer
from compact text alone, especially in refusal, no-tool, or soft-authorization
cases.

The tradeoff is slightly worse bad-tool exposure:

```text
bad3: .020 -> .030
bad5: .040 -> .050
```

This should be monitored before production use, but it is not severe enough to
discard the feature group at this stage.

Conclusion:

```text
Interaction features are the best candidate for the next controlled model.
```

### 5.5 `cross-only` helps top3 but weakens overall ranking quality

`cross-only` improves top3:

```text
top3: .900 -> .910
```

but hurts:

```text
ndcg5, regret5, bad3, bad5, MAE, Spearman
```

This indicates that coarse multiplicative signals can help get the best tool
into the candidate shortlist, but they are not reliable enough for score
ordering or calibration.

Conclusion:

```text
Cross features may be useful as weak auxiliary features, but should not be the
main policy feature path.
```

## 6. Main Conclusion

The ablation does not support "add all policy features directly" as the next
model. The useful part is narrower:

```text
Keep interaction features.
Treat conv features as weak calibration/gating.
Avoid direct tool-only static priors in the ranker.
Treat cross features cautiously.
```

The most defensible next baseline is:

```text
V2.2b Lambda + interaction-only policy features
```

It is the best direct ranking tradeoff:

```text
top1 improves
top3 nearly preserved
regret preserved
ndcg preserved
MAE slightly improves
model remains lightweight
```

## 7. Recommended Next Experiments

### 7.1 Interaction-only confirmatory run

Run one more controlled variant:

```text
V2.2b Lambda + interaction-only + lower policy dropout or feature dropout
```

Goal:

```text
Keep top1 gain while recovering bad3/bad5.
```

### 7.2 Weak conv fusion

Test:

```text
interaction + scaled conv
```

where conv features are multiplied by a small scalar, for example:

```text
conv_scale = 0.25 or 0.5
```

Rationale:

Conv-only improves high-value recall and precision, but hurts precise ranking.
It should contribute weak global context rather than dominate pair ranking.

### 7.3 Remove tool-only from direct ranker

Do not include `tool` features in the main ranking side channel for the next
baseline.

If tool features are reused later, prefer one of:

```text
separate risk/calibration head
heavy feature dropout
small learned gate initialized near zero
offline analysis only
```

### 7.4 Add bad-case-specific evaluation slices

The aggregate metrics hide important N.E.K.O-specific failures. The next report
should track slices such as:

```text
no-tool / companion / emotional support
explicit refusal or boundary
agentic task
external side-effect tools
monitoring or proactive tools
file/browser/search tools
```

This matters because a small top-k gain can be unacceptable if it increases
tool activation in emotional-support or refusal contexts.

## 8. Practical Recommendation

For the next mainline model, use:

```text
base:          V2.2b Lambda
extra feature: interaction-only
avoid:         tool-only direct feature concat
watch:         bad_top3_rate, bad_top5_rate, no-tool false positives
```

This keeps the model lightweight and aligned with the current research goal:
continuous confidence reranking for personalized N.E.K.O tool recommendation,
without turning the system into a pure task router.
