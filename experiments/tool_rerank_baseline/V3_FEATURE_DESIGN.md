# V3.1 Metadata-Light Policy Residual Design

Date: 2026-05-30

V3.1 targets P0 personalized plugin recommendation/control for N.E.K.O while
preserving the V2.2b ranking backbone. The design is based on the V2.2b policy
feature ablation: policy signals are useful, but directly concatenating all of
them into the main ranking side-channel hurts precise top-k ranking.

## Constraints

- Keep grouped 5-fold CV as the only training validation mode.
- Keep the frozen Jina sequence path unchanged.
- Do not require richer plugin metadata.
- Do not add extra Jina computation.
- Do not introduce discrete action labels.
- Keep train and inference close to linear cost.
- Preserve continuous confidence prediction.

## Inputs

The feature extractor uses only data already available in the current dataset:

- `conversation_text`
- `scenario_type`
- `tool_relevance_mode`
- `tool_id`
- rendered `tool_text`

`raw_score` is intentionally excluded from policy features because it is a
teacher-side signal and would leak label construction information.

## Feature Groups

`policy_features.py` currently exposes four metadata-light feature groups:

```text
conv          18 features
tool          20 features
interaction   11 features
cross          8 features
```

The ablation result suggests the following roles:

```text
interaction  main policy residual
conv         weak gate / weak bias
tool         not used directly in main ranker
cross        optional later experiment
```

## Architecture

V3.1 keeps the V2.2b semantic and lexical backbone:

```text
semantic_features = Jina sequence late interaction(conversation, tool)
lexical_delta     = gated BM25/ngram projection
base_score        = MLP(semantic_features + lexical_delta)
```

Policy is added as a bounded residual, not as an unconstrained concatenated
ranking feature:

```text
policy_delta = policy_scale * tanh(policy_head(interaction_features))
```

When conversation gate features are enabled:

```text
policy_multiplier = 2 * sigmoid(conv_gate(conv_features))
conv_bias         = conv_bias_scale * tanh(conv_bias_head(conv_features))

final_score = base_score + policy_multiplier * policy_delta + conv_bias
```

The default residual scale is intentionally small:

```text
policy_scale = 0.3
conv_bias_scale = 0.05
```

This makes policy a correction to the semantic/lexical ranker rather than a
replacement for it.

## Why Not Full Policy Concat

The V2.2b ablation showed:

```text
policy_all:  top1 .800 -> .750, top3 .900 -> .850
tool-only:   top3 .900 -> .830, regret5 .033 -> .069
interaction: top1 .800 -> .830, regret5 preserved
```

Full policy concat improves broad calibration metrics, but it hurts precise
candidate-set ranking. Tool-only features behave like an overconfident static
prior: they can make the model more confident without making it better at
selecting the right tool for the current user turn.

For N.E.K.O this is risky because the system is not a pure tool router. A user
mentioning a domain does not necessarily authorize action, monitoring, upload,
messaging, browsing, or agent control.

## Active V3.1 Variants

The first V3.1 CV configs are:

```text
config_jina_sequence_policy_v31_interaction_residual_cv_lambda_gpu.toml
config_jina_sequence_policy_v31_interaction_conv_gate_cv_lambda_gpu.toml
config_jina_sequence_policy_v31_interaction_conv_gate_dropout_cv_lambda_gpu.toml
```

They test:

```text
A. interaction-only bounded residual
B. interaction residual + weak conv gate/bias
C. interaction residual + weak conv gate/bias + stronger feature dropout
```

All three keep the V2.2b LambdaNDCG setup fixed.

## Success Criteria

Compared with `jina_sequence_lexical_gated_v22b_cv_lambda_gpu`, a useful V3.1
variant should aim for:

```text
top1_match       >= 0.83
topk_recall      >= 0.89
top5_regret      <= 0.033
bad_top3_rate    close to baseline
bad_top5_rate    close to baseline
no_tool_fp_rate  no regression
```

Slice metrics should be added before treating the result as stable:

```text
companion / emotional support / no-tool
boundary or refusal
agentic task
explicit plugin action
external side-effect tools
monitoring or proactive tools
file/browser/search tools
```
