# V2.3 Action-Consent Feature Report

Date: 2026-05-31

## Summary

V2.3 tested a lightweight feature-design change on top of the V2.2b LambdaNDCG
baseline. The goal was to stay within the V2 parameter budget: no V4 factors,
no new transformer, no larger head, and no plugin-specific rules.

The experiment added one metadata-light policy feature group:

```text
action_consent: 45 features
```

These features try to preserve action/consent evidence that coarse V2.2 policy
features lose:

- no-call / ask-confirm / can-call intent;
- explicit refusal and temporary defer;
- denied operation scopes such as search, monitoring, upload, message, control,
  and file access;
- requested operation scopes such as browser, form fill, search, save template,
  message reply, file access, monitoring, and device control;
- tool side-effect/risk abstractions;
- requested-operation matches and denied-operation conflicts.

The change was run on the remote GPU machine:

```text
host: connect.westb.seetacloud.com:47991
project: /autodl-fs/data/playground4
gpu: NVIDIA GeForce RTX 4080 SUPER, 32GB
python: 3.12.3
torch: 2.8.0+cu128
```

## Configuration

Base:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_gated_v22b_cv_lambda_gpu.toml
```

V2.3:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v23_action_consent_cv_lambda_gpu.toml
```

Run:

```text
experiments/tool_rerank_baseline/runs/jina_sequence_lexical_policy_v23_action_consent_cv_lambda_gpu
```

Common setup:

```text
dataset:       data/runs/neko_v3_500
validation:    grouped 5-fold CV
holdout:       20% conversation groups excluded from CV
encoder:       jinaai/jina-embeddings-v5-text-nano-retrieval
model:         sequence_late_interaction
head:          MLP
lexical:       gated_add
loss:          0.7 SmoothL1 + 0.3 LambdaNDCG@5
epochs:        8
```

V2.3 only adds:

```toml
[policy_features]
enabled = true
groups = ["action_consent"]
```

The feature group is concatenated into the existing lexical side channel. It
does not add a residual head or extra model branch.

## Aggregate Results

5-fold CV mean:

```text
metric                  v22b_lambda   v22b_interaction   v31_gate_dropout   v23_action_consent
top1_match              0.8000        0.8300             0.8000             0.8000
topk_recall             0.9000        0.8900             0.9000             0.9300
ndcg_at_5               0.8860        0.8868             0.8897             0.8946
top5_regret             0.0332        0.0332             0.0407             0.0323
bad_top3_rate           0.0200        0.0300             0.0200             0.0200
bad_top5_rate           0.0400        0.0500             0.0300             0.0500
no_tool_fp_rate         0.0000        0.0000             0.0000             0.0000
high_conf_precision     0.7183        0.7479             0.8302             0.8163
high_value_recall_at_3  0.5480        0.5532             0.5367             0.5949
high_value_recall_at_5  0.7131        0.7081             0.7251             0.7668
mae                     0.5655        0.5612             0.5662             0.5537
spearman                0.6968        0.6956             0.6987             0.7052
```

Delta against V2.2b Lambda:

```text
top1_match              +0.0000
topk_recall             +0.0300
ndcg_at_5               +0.0086
top5_regret             -0.0009
bad_top3_rate           +0.0000
bad_top5_rate           +0.0100
no_tool_fp_rate         +0.0000
high_conf_precision     +0.0980
high_value_recall_at_3  +0.0469
high_value_recall_at_5  +0.0537
mae                     -0.0118
spearman                +0.0083
```

## Interpretation

V2.3 is a real improvement for broad ranking and calibration:

- Top3 recall improves from `0.9000` to `0.9300`.
- NDCG@5 improves from `0.8860` to `0.8946`.
- High-confidence precision improves from `0.7183` to `0.8163`.
- High-value recall improves at both top3 and top5.
- MAE and Spearman improve.

But it is not a clean solution:

- Top1 remains unchanged at `0.8000`.
- Bad top5 exposure worsens from `0.0400` to `0.0500`.
- Several newly regressed top1 cases show that positive match features can
  over-lift broad operation/tool-effect matches.

In short:

```text
V2.3 improves recall and calibration, but does not solve first-rank precision
or boundary-sensitive bad exposure.
```

## Bad Case Summary

Same CV prediction files were compared across V2.2b Lambda, V2.2b interaction,
and V2.3 action-consent.

Counts:

```text
run                 top1_miss  top3_miss  top3_bad_items  top5_bad_items  fp  fn
v22b_lambda         20         10         2               6               15  24
v22b_interaction    17         11         4               6               11  35
v23_action_consent  20         7          3               7               10  20
```

V2.3 fixes some top3 and false-positive/false-negative behavior, but the top1
miss count does not move.

### Fixed vs V2.2b

Top1 fixed:

```text
conv_000065
conv_000234
conv_000311
conv_000320
conv_000365
conv_000427
conv_000487
```

Top3 fixed:

```text
conv_000320
conv_000365
conv_000408
conv_000459
```

Important positive sign:

- `conv_000320`, an explicit browser/form-fill task, improves. This suggests
  requested-operation features are useful for actionable browser tasks.

### Regressed vs V2.2b

Top1 regressed:

```text
conv_000033
conv_000119
conv_000232
conv_000264
conv_000284
conv_000398
conv_000415
```

Top3 regressed:

```text
conv_000264
```

Top5 bad exposure regressed:

```text
conv_000258 / plugin.slack_auto_reply
conv_000322 / plugin.security_camera_events
```

### Representative Remaining Failures

`conv_000322` remains the most important boundary failure:

```text
scenario: boundary_or_refusal_no_tool / no_tool_or_low_relevance
V2.3 top1: plugin.emotion_lamp_listener
gold:      plugin.study_companion
```

The user explicitly rejects emotion monitoring / calming-light reminders, but
the model still ranks the emotion-lamp tool first. V2.3 does create monitoring
denial/conflict features, but the learned score does not treat them as a hard
veto. This case is also affected by label shape: the emotion-lamp candidate is
not strongly negative in the V2 data, so the model is not punished enough for
keeping it high.

New regressions such as `conv_000119`, `conv_000264`, and `conv_000284` suggest
that broad positive operation matches can over-lift tools with generic home,
ritual, calendar, or device-control language.

## Design Lesson

V2.3 proves that action-consent information is useful, but the feature direction
is imbalanced:

```text
positive match features are too strong;
negative conflict/veto features are too weak.
```

Because plugins are an open, variable class, the next design should not target
specific plugin IDs or plugin categories. It should stay at a stable behavior
abstraction layer:

```text
What operation does the user authorize?
What operation does the tool perform?
What side effect would the tool cause?
Does the current conversation deny or defer that operation?
```

The key distinction is:

```text
do not identify the plugin;
identify the consequences of invoking it.
```

## Recommendation For V2.3b

Keep the V2.2b backbone and stay lightweight, but reduce the feature set and
make directionality clearer.

### Keep

Conversation state:

```text
no_call_intent
ask_confirm_intent
can_call_intent
temporary_defer
final_confirmation_required
```

Tool effects:

```text
read_only
local_write
external_action
background_monitor
user_state_inference
device_or_environment_control
agentic_operation
commit_or_submit
```

Pair evidence:

```text
weak_operation_match
strong_operation_match
denied_operation_conflict
no_call_side_effect_conflict
confirmation_boundary_conflict
```

### Cut Or Weaken

Features that can become broad positive priors:

```text
requested_device_control
match_requested_device_control
requested_monitoring
match_requested_monitoring
tool_home / tool_audio / tool_emotion style taxonomy
```

These are too close to plugin category priors. In an open plugin universe, they
can over-lift unrelated tools with similar atmosphere/device language.

### Principle

Use feature design to preserve decision evidence, not to classify plugins:

```text
match = weak positive evidence
conflict = strong negative evidence
veto = strongest negative evidence
```

V2.3b should therefore be smaller than V2.3, not larger.

## Post-Hoc Calibration Probe

After V2.3b underperformed V2.3, we tested a no-retrain post-hoc calibration
probe on top of the V2.3 prediction files. The probe uses only existing
action-consent evidence and subtracts lightweight penalties from candidates with
side effects, denied operations, refusal vetoes, or confirmation-boundary
conflicts.

The best aggressive setting found in the first scan was:

```text
side_effect_penalty       0.20
denied_operation_penalty  0.30
refusal_veto_penalty      0.40
confirmation_penalty      0.20
strong_match_bonus        0.00
weak_match_bonus          0.00
```

Approximate CV metrics on V2.3 predictions:

```text
metric              v23_base  aggressive_posthoc
top1_match          0.8000    0.7600
top3_recall         0.9300    0.9300
bad_top3_rate       0.0200    0.0000
bad_top5_rate       0.0500    0.0100
high_conf_precision 0.7962    0.8800
mae                 0.5537    0.5724
spearman            0.7029    0.6990
```

This confirms the feature evidence is directionally useful: conflict/veto
signals can sharply reduce bad exposure. But the trade-off is too expensive for
ranking: top1 drops by four points and calibration worsens.

A milder operating point is more realistic:

```text
side_effect_penalty       0.10
denied_operation_penalty  0.20
refusal_veto_penalty      0.30
confirmation_penalty      0.05
strong_match_bonus        0.00
weak_match_bonus          0.00
```

Approximate CV metrics:

```text
metric              v23_base  mild_posthoc
top1_match          0.8000    0.7900
top3_recall         0.9300    0.9300
bad_top3_rate       0.0200    0.0100
bad_top5_rate       0.0500    0.0300
high_conf_precision 0.7962    0.8467
mae                 0.5537    0.5565
spearman            0.7029    0.7054
```

This is a useful diagnostic but not yet a final solution. It says the missing
piece is not another V4-style model branch. The missing piece is a lightweight
activation/ceiling layer:

```text
ranking score answers: which tool is best if a tool should be used?
activation score answers: should this candidate be allowed to surface at all?
```

The current V2 objective is still forced-pick within each conversation group.
When a no-call or refusal conversation has many mildly positive labels, the
ranker learns to put some tool first. Feature penalties can suppress obvious
bad exposure, but without an explicit activation ceiling they either remain too
weak or start damaging legitimate top1 matches.

Recommended next step:

```text
Keep V2.3 as the main trained model.
Do not adopt V2.3b as mainline.
Add a small post-ranker activation/ceiling stage and tune it on CV predictions.
```

The activation stage should stay model-light: a few scalar rules or a linear
calibrator over behavior-level signals, not an MLP and not plugin-specific
logic.
