# V4 Model Design Plan

## Background

The current v2.2b/v3 line already proves that a lightweight reranker can learn
usable tool preference from N.E.K.O conversations. However, recent experiments
also show that most improvements are local: lexical features, policy features,
and gated fusion improve some metrics, but do not fundamentally solve the
hardest ambiguity.

The core issue is label compression. A single `confidence` target mixes several
different decision factors:

- whether the tool can do the task;
- whether the user actually wants action;
- whether the request points to this specific tool category;
- whether the user has given enough permission;
- whether the tool is too intrusive or side-effectful;
- whether tool use fits N.E.K.O's companionship moment.

For a companionship-first agent, these factors are not interchangeable.
Semantic match is not final recommendation. Capability is not consent. High
cost is not always bad. A tool can be useful while still inappropriate in the
current emotional moment.

V4 should therefore use the v4 multi-axis dataset as supervision instead of
only learning final preference directly.

## Objective

Build a lightweight, fast, research-friendly V4 reranker that predicts:

1. Six factor scores:
   - `capability_match`
   - `action_demand`
   - `target_specificity`
   - `consent_boundary`
   - `intervention_cost`
   - `companionship_fit`
2. One final score:
   - `final_preference`

The model should remain cheap enough for 5-fold CV and fast iteration. The
initial goal is not maximum architecture complexity. The goal is to prove
whether factorized supervision creates a cleaner decision space than the current
single-label baseline.

## Dataset Assumptions

Two dataset lines should be kept separate:

- `neko_v3e_500`: refreshed single-score LLM-B dataset. This is useful as the
  strongest v3 single-target baseline.
- v4 multi-axis dataset: required for the actual V4 model. It must contain
  `axis_scores` and keep `confidence = axis_scores.final_preference` for
  backward compatibility.

V4 experiments should not silently train on v3e as if it were multi-axis data.
If `axis_scores` is missing, the experiment should either fail clearly or run
only a v3-compatible baseline mode.

## Architecture

### Inputs

Reuse the stable v2.2b input channels:

- conversation text;
- tool `source_text`;
- Jina semantic representation;
- lexical BM25/ngram features;
- minimal tool identity/kind metadata;
- optional scenario/provenance hints from generated conversations.

Do not add a heavy online text encoder at this stage. Jina features should stay
cached. V4 should focus on the decision head, not on making embedding
infrastructure more expensive.

### Shared Pair Encoder

The shared encoder should stay close to v2.2b:

```text
conversation/tool semantic features
+ lexical features
+ minimal metadata features
-> shared_pair_feature
```

Candidate implementation:

```text
semantic_projection: Linear/MLP
lexical_projection: small Linear
lexical_gate: sigmoid(gate_input)
shared_pair_feature = semantic_feature + lexical_gate * lexical_projection
```

This keeps the useful v2.2b behavior while preventing lexical features from
becoming the whole model.

### Factor Heads

Use one small head per factor:

```text
shared_pair_feature -> factor_heads -> 6 factor predictions
```

Each factor prediction is continuous and should live in the same latent space as
the calibrated labels.

The factor heads can share a hidden layer:

```text
factor_hidden = MLP(shared_pair_feature)
factor_pred = Linear(factor_hidden, 6)
```

This is still lightweight and faster than independent deep heads.

### Final Preference Head

The final score should not be predicted only from raw shared features. It should
explicitly receive the factor predictions.

Initial design:

```text
factor = [
  capability_match,
  action_demand,
  target_specificity,
  consent_boundary,
  intervention_cost,
  companionship_fit,
]

interaction = [
  capability_match * action_demand,
  capability_match * target_specificity,
  action_demand * consent_boundary,
  consent_boundary * companionship_fit,
  action_demand * companionship_fit,
  intervention_cost * action_demand,
  intervention_cost * consent_boundary,
]

final_input = concat(
  small_projection(shared_pair_feature),
  factor,
  interaction,
)

final_preference = Linear or small MLP(final_input)
```

Important: `intervention_cost` is not an inverse score. High cost means the tool
is disruptive, external, privacy-sensitive, proactive, or side-effectful. Its
effect depends on action demand and consent. The model should learn this through
interactions instead of a hard negative coefficient.

## Loss Design

Start with a conservative multi-task objective:

```text
loss =
  final_regression
+ alpha * factor_regression
+ beta  * final_ranking
+ gamma * soft_safety_consistency
```

### Final Regression

Use SmoothL1:

```text
SmoothL1(final_pred, final_preference_label)
```

This preserves the current continuous target behavior and is robust to noisy
LLM-B scores.

### Factor Regression

Use mean SmoothL1 across the six factor axes:

```text
mean SmoothL1(factor_pred_i, factor_label_i)
```

Initial weight:

```text
alpha = 0.2 to 0.5
```

If factor heads dominate and hurt ranking, lower `alpha`. If final preference
still looks noisy, raise `alpha`.

### Final Ranking

Keep the ranking loss on `final_preference`, not on every factor.

Candidates:

- LambdaNDCG, matching the current strongest training direction;
- pairwise margin loss inside each candidate set.

Initial recommendation:

```text
beta = 0.2 to 0.5
```

Ranking should improve top-k behavior without destroying calibrated regression.

### Soft Safety Consistency

This should be small and diagnostic, not a hard rule engine.

Examples:

```text
low consent_boundary should penalize very high final_pred
low action_demand should penalize very high final_pred
low capability_match should penalize very high final_pred
```

One possible form:

```text
penalty = relu(final_pred - threshold) * sigmoid(-factor_pred)
```

Initial recommendation:

```text
gamma = 0.02 to 0.05
```

This term should be ablated. If it reduces bad cases but hurts top-k, keep it
optional.

## Metrics

V4 should report both final ranking metrics and factor quality metrics.

### Final Preference Metrics

Use existing metrics:

- Top1 / Top3 / Top5;
- NDCG@3 / NDCG@5;
- regret@3 / regret@5;
- bad@3 / bad@5;
- high-confidence precision;
- MAE / MSE;
- Spearman correlation.

Add N.E.K.O-specific slice metrics:

- low-action false positive rate;
- low-consent false positive rate;
- high-cost false positive rate;
- companionship-only/no-tool false positive rate;
- assistant-suggested-but-not-accepted false positive rate.

### Factor Metrics

For each factor:

- MAE;
- Spearman;
- per-scenario MAE;
- factor calibration histogram;
- factor-vs-final correlation.

The factor-vs-final correlation is important. If all factors collapse into the
same curve, the dataset did not become orthogonal enough.

## Ablation Plan

Run all experiments under the existing 5-fold CV protocol.

### A. V3E Final-Only Baseline

Train the current v2.2b model on `neko_v3e_500`.

Purpose:

- establish the strongest single-label baseline after LLM-B rerun;
- separate dataset refresh gains from architecture gains.

### B. V4 Final-Only Baseline

Train v2.2b-style model on v4 data using only:

```text
confidence = axis_scores.final_preference
```

Purpose:

- measure whether v4 data itself improves final labels even without factor
  heads.

### C. V4 Auxiliary Factor Heads

Train:

```text
shared_pair_feature -> final head
shared_pair_feature -> factor heads
```

But final head does not consume factor predictions.

Purpose:

- test whether factor supervision regularizes the shared representation.

### D. V4 Factor-To-Final Linear Head

Train:

```text
shared_pair_feature -> factor heads
factor predictions -> linear final head
```

Purpose:

- test the user's hypothesis: factors are high-order features and final
  preference can be learned by another linear layer.

### E. V4 Factor Interactions

Train:

```text
factor predictions + selected pairwise interactions -> final head
```

Purpose:

- test whether final recommendation depends on non-additive relations such as:
  consent x cost, action x cost, companionship x action.

### F. V4 Ranking Loss

Add LambdaNDCG or pairwise ranking loss on final preference.

Purpose:

- improve top-k recommendation without losing continuous calibration.

### G. V4 Consistency Penalty

Add soft safety consistency.

Purpose:

- reduce high-confidence mistakes in low-consent / low-action / no-tool cases.

## Expected Outcomes

The minimum useful V4 result is not simply higher Top1.

The model is promising if:

- Top3/Top5 remain strong or improve;
- bad@3/bad@5 drop;
- low-consent and low-action false positives drop;
- factor predictions show non-trivial Spearman correlation;
- factor outputs do not collapse into near-identical scores;
- final preference remains calibrated enough for thresholding.

The model is not promising if:

- factor heads add loss but final metrics are unchanged;
- all factors become highly collinear;
- final head ignores factor predictions;
- high-cost or low-consent bad cases remain unchanged.

## Implementation Plan

### Step 1: Data Loader Upgrade

Add optional multi-axis label loading:

```text
PairExample.label              -> final confidence
PairExample.axis_labels         -> dict or fixed vector of six factor labels
PairExample.final_axis_label    -> final_preference
```

Requirements:

- old v3/v3e data still loads;
- v4 mode fails clearly if `axis_scores` is missing;
- axis order is fixed and centralized.

### Step 2: Model Config

Add model variants:

```text
v4_final_only
v4_aux_factors
v4_factor_linear_final
v4_factor_interactions
```

Avoid too many command-line flags. Use config files so CV runs are reproducible.

### Step 3: Loss Module

Implement:

- final SmoothL1;
- factor SmoothL1;
- optional LambdaNDCG/pairwise ranking;
- optional consistency penalty.

Log each loss component separately every epoch.

### Step 4: Metrics Module

Extend metrics to include:

- factor MAE/Spearman;
- low-action false positive rate;
- low-consent false positive rate;
- high-cost false positive rate;
- no-tool/companionship slice metrics.

### Step 5: CV Runner

Keep mandatory 5-fold CV.

The runner should output:

- per-fold metrics;
- mean/std metrics;
- per-epoch curves;
- plots;
- predictions for bad-case inspection.

### Step 6: Report

Write a V4 early report after A-D at minimum.

The report should answer:

- Does v4 data improve over v3e single labels?
- Do factor heads regularize the model?
- Can factor predictions linearly explain final preference?
- Which slices improve or degrade?
- Are remaining failures data issues, feature issues, or model issues?

## Risk Notes

### Axis Non-Orthogonality

The axes may still be correlated because LLM-B may not truly separate them.
This is not fatal, but it must be measured. If factor correlations are too high,
dataset prompt engineering is the next lever, not model complexity.

### Label Noise

LLM-B factor labels are still synthetic judgments. Do not overfit the model to
small numeric differences between adjacent tools. Ranking and calibration should
matter more than exact point regression.

### Over-Regularization

Consistency penalties can accidentally turn the model into a rule system.
They should stay small and ablated.

### Cost

V4 model training should remain cheap. The expensive step is Jina/cache and
LLM-B data generation, not the decision head. Do not introduce an online large
text encoder until the factorized supervision shows clear value.

## First Milestone

The first serious milestone is:

```text
V3E final-only baseline
V4 final-only baseline
V4 auxiliary factor heads
V4 factor-to-final linear head
V4 factor interactions
```

All under 5-fold CV, with a short but rigorous report comparing:

- final ranking;
- continuous calibration;
- factor prediction quality;
- low-action/low-consent bad cases;
- qualitative top-k examples.

Only after this milestone should we consider heavier architecture changes.
