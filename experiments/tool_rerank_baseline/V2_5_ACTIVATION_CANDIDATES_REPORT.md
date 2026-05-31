# V2.4-V2.5b Activation Candidate Report

Date: 2026-05-31

## Summary

This round stays on the V2.2/V2.3 line. It does not use the V4 feature stack,
does not add a larger head, and does not introduce plugin-specific rules.

The key change is to stop treating activation policy only as side-channel
features. Instead, V2.4/V2.5 add explicit virtual candidates into the same
ranking list:

```text
V2.4   __NO_TOOL__
V2.5   __NO_TOOL__ + __ASK_CONFIRM__
V2.5b  V2.5 plus a label ceiling for real tools in ask-confirm groups
```

This changes the problem from:

```text
"rank the best concrete plugin, while implicitly learning whether to call"
```

to:

```text
"rank CALL_TOOL / ASK_CONFIRM / NO_TOOL in the same candidate set"
```

The design is still lightweight. The encoder, lexical side channel, MLP head,
loss, optimizer, CV split, and training budget remain the same as V2.3.

## Motivation

V2.3 action-consent features helped recall and calibration, but did not solve
the core boundary problem:

- top1 stayed at `0.8000`;
- raw bad top5 exposure worsened from `0.0400` to `0.0500`;
- positive operation-match features could still over-lift broad but unsafe or
  premature tools.

The failure pattern suggested that the model needed an explicit alternative to
calling a concrete plugin. If the only high-scoring objects are real plugins,
weak authorization and no-tool cases must be represented indirectly by lowering
all real tools. That is hard for a small regressor and unstable across plugin
classes.

## Configurations

Base V2.3:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v23_action_consent_cv_lambda_gpu.toml
```

New configs:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v24_no_tool_candidate_cv_lambda_gpu.toml
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v25_ask_confirm_candidate_cv_lambda_gpu.toml
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v25b_ask_confirm_ceiling_cv_lambda_gpu.toml
```

Common setup:

```text
dataset:       data/runs/neko_v3_500
validation:    grouped 5-fold CV
holdout:       20% conversation groups excluded from CV
encoder:       jinaai/jina-embeddings-v5-text-nano-retrieval
model:         sequence_late_interaction
head:          MLP, hidden_dim 256
lexical:       gated_add
policy:        action_consent
loss:          0.7 SmoothL1 + 0.3 LambdaNDCG@5
epochs:        8
```

## Aggregate Results

5-fold CV mean:

```text
metric                      v23       v24       v25       v25b
top1_match                  .8000     .8600     .8800     .9000
topk_recall                 .9300     .9700     .9900     .9900
ndcg_at_5                   .8946     .9242     .9179     .9208
bad_top3_rate               .0200     .0200     .0000     .0100
bad_top5_rate               .0500     .0300     .0000     .0200
bad_exposed_top1_rate       n/a       .0000     .0000     .0000
bad_exposed_top3_rate       n/a       .0000     .0000     .0000
bad_exposed_top5_rate       n/a       .0000     .0000     .0000
activation_accuracy         n/a       .9600     n/a       .9900
activation_action_accuracy  n/a       n/a       .9300     .9500
no_tool_recall              n/a       .9633     1.0000    1.0000
no_tool_precision           n/a       .9550     .9500     .9833
ask_confirm_recall          n/a       n/a       .7870     .8425
ask_confirm_precision       n/a       n/a       1.0000    1.0000
active_tool_top1            n/a       .7538     .7673     .8028
active_tool_topk            n/a       .9600     .9778     .9778
high_conf_precision         .8163     .8914     n/a       .8178
```

## Metric Note: Raw Bad vs Exposed Bad

After introducing virtual action candidates, raw `bad_top3_rate` and
`bad_top5_rate` can be misleading. A bad real plugin may appear below
`__NO_TOOL__` or `__ASK_CONFIRM__`, but if the runtime action is not a tool
call, that plugin is not actually exposed to the user.

Therefore this round adds exposure-aware metrics:

```text
if top1 is __NO_TOOL__ or __ASK_CONFIRM__, exposed top1/top3/top5 are empty
```

Under this metric, V2.4/V2.5/V2.5b all reach:

```text
bad_exposed_top1_rate = 0.0000
bad_exposed_top3_rate = 0.0000
bad_exposed_top5_rate = 0.0000
```

This is the most important positive result of the activation-candidate design.

## Interpretation

V2.4 proves that adding `__NO_TOOL__` is the right shape for no-call cases:

- top1 improves from `.8000` to `.8600`;
- top3 improves from `.9300` to `.9700`;
- NDCG@5 improves from `.8946` to `.9242`;
- exposed bad tools drop to zero.

V2.5 extends the same idea to weak authorization:

- top1 improves to `.8800`;
- top3 reaches `.9900`;
- ask-confirm precision is perfect, but recall is only `.7870`.

V2.5b adds a conservative real-tool label ceiling for ask-confirm groups:

- top1 improves again to `.9000`;
- action accuracy improves to `.9500`;
- ask-confirm recall improves to `.8425`;
- ask-confirm precision remains `1.0000`;
- active concrete-tool top1 also improves to `.8028`.

This is a stronger result than the earlier feature-only attempts because the
model no longer has to encode "do not call" only as a negative bias over every
plugin. It can place a policy action above the real tools.

## Remaining Bad Cases

V2.5b still has 5 action errors:

```text
conv_000151  passive_event actionable_tool_relevance       gold ASK_CONFIRM, pred plugin.tabletop_soundscape
conv_000249  passive_event actionable_tool_relevance       gold ASK_CONFIRM, pred plugin.arxiv_research_finder
conv_000259  passive_event actionable_tool_relevance       gold ASK_CONFIRM, pred plugin.group_costume_vote_host
conv_000278  ambiguous_need_clarification weak_or_requires_confirmation  gold ASK_CONFIRM, pred plugin.paper_research_assistant
conv_000441  assistant_suggested_tool_no_auth no_tool_or_low_relevance   gold ASK_CONFIRM, pred __NO_TOOL__
```

The dominant residual slice is not ordinary no-tool detection. It is the
middle action:

```text
the conversation mentions a plausible tool domain, but authorization is weak
or only passively implied
```

The non-action top1 misses are mostly exact tool-family ordering errors:

```text
computer/browser/qwen/openfang family
obsidian/notebook_lm family
research-agent variants
```

These are less dangerous than activation errors because exposed bad rate is
already zero, but they limit top1.

## Conclusion

V2.5b is the current best V2-line mainline.

The lesson is that feature design alone was too implicit. The useful lightweight
abstraction is an action-level candidate:

```text
NO_TOOL / ASK_CONFIRM / CALL_TOOL
```

This keeps the model small while giving the ranker a direct object to choose
when the correct behavior is not a plugin call.

The next experiment should focus narrowly on the ask-confirm residual slice,
especially `passive_event` and weak authorization. The safest first check is a
V2.5c label-strength test: increase ask-confirm margin and real-tool ceiling
pressure without changing architecture. If that helps recall without hurting
precision, the remaining issue is label strength. If it does not, the problem is
semantic ambiguity and needs a better weak-authorization feature design rather
than more margin.

## Follow-up: V2.5c and V2.5d

Two follow-up checks were run after V2.5b.

### V2.5c: Stronger Ask-Confirm Margin

Config:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v25c_ask_confirm_strong_cv_lambda_gpu.toml
```

Change:

```text
ask_confirm.label_margin:          0.50 -> 0.75
ask_confirm.real_tool_ceiling:     0.75 -> 1.00
```

Result:

```text
metric                      v25b      v25c
top1_match                  .9000     .8900
topk_recall                 .9900     .9900
ndcg_at_5                   .9208     .9239
activation_accuracy         .9900     .9700
activation_action_accuracy  .9500     .9300
no_tool_recall              1.0000    1.0000
no_tool_precision           .9833     .9262
ask_confirm_recall          .8425     .7981
ask_confirm_precision       1.0000    1.0000
active_tool_top1            .8028     .7984
bad_exposed_top5_rate       .0000     .0000
```

Interpretation:

```text
More pressure on ask-confirm does not fix the residual ask-confirm -> call-tool
errors. It also introduces extra ask-confirm -> no-tool confusion.
```

So the remaining problem is not simple label strength.

### V2.5d: Latest-User Granularity

Config:

```text
experiments/tool_rerank_baseline/config_jina_sequence_lexical_policy_v25d_latest_user_action_cv_lambda_gpu.toml
```

Change:

- preserve `authorization_level`, `latest_user_actionability`, and
  `latest_user_text` in `PairExample`;
- make action-consent features prefer the latest user turn instead of the full
  conversation for authorization/refusal/request signals;
- exclude no-tool groups from ask-confirm labels;
- exclude latest-user actionable groups from ask-confirm labels.

This keeps the feature dimension unchanged:

```text
policy feature_dim = 45
```

Result:

```text
metric                      v25b      v25d
top1_match                  .9000     .8400
topk_recall                 .9900     .9300
ndcg_at_5                   .9208     .9016
activation_accuracy         .9900     .9000
activation_action_accuracy  .9500     .8700
no_tool_recall              1.0000    .8810
no_tool_precision           .9833     .9652
ask_confirm_recall          .8425     .6433
ask_confirm_precision       1.0000    .6833
active_tool_top1            .8028     .7917
active_tool_topk            .9778     .9167
bad_exposed_top5_rate       .0000     .0000
```

V2.5d fixed the original V2.5b residual action errors for several cases:

```text
conv_000151  ASK_CONFIRM -> CALL_TOOL became CALL_TOOL -> CALL_TOOL
conv_000249  ASK_CONFIRM -> CALL_TOOL became NO_TOOL -> NO_TOOL
conv_000259  ASK_CONFIRM -> CALL_TOOL became CALL_TOOL -> CALL_TOOL
conv_000278  ASK_CONFIRM -> CALL_TOOL became ASK_CONFIRM -> ASK_CONFIRM
conv_000441  ASK_CONFIRM -> NO_TOOL became NO_TOOL -> NO_TOOL
```

But it created more new boundary errors:

```text
v25b action errors: 5
v25d action errors: 13
```

Interpretation:

```text
Latest-user granularity is useful, but hard label rewriting is too brittle.
```

The information should be used as a soft feature or auxiliary action signal,
not as a direct replacement for the original V2.5b virtual-candidate labels.

## Current Recommendation

Keep V2.5b as the mainline.

Do not continue with V2.5c-style stronger margins, and do not promote V2.5d's
hard label exclusions. The next useful direction is a V2.6-style soft action
prior:

```text
base labels:        keep V2.5b
features:           add/latest-user action evidence
training target:    optionally add a small auxiliary action loss
parameters:         keep the same sequence model and small head
```

The core idea is to let latest-user authorization/refusal explain boundary
cases without deleting the broader weak-authorization supervision that V2.5b
needed for recall.
