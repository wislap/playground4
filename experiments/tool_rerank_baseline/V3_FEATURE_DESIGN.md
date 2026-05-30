# V3 Metadata-Light Policy Features

V3 targets P0 personalized plugin recommendation/control for N.E.K.O. It keeps
the frozen Jina sequence path unchanged and adds a small continuous side-channel
for permission, boundary, and plugin-control policy.

## Constraints

- Do not require richer plugin metadata.
- Do not add extra Jina computation.
- Do not introduce discrete action labels.
- Keep train/inference close to linear cost.
- Support 5-fold CV only.

## Inputs

The feature extractor uses only data already available in the current dataset:

- `conversation_text`
- conversation provenance rendered into text
- `scenario_type`
- `tool_relevance_mode`
- `tool_id`
- rendered `tool_text`
- `raw_score`

## Feature Groups

`policy_features.py` emits 63 continuous features:

- conversation policy features:
  `agentic_task`, `companion`, `boundary`, `actionable`, `authorized`,
  explicit refusal/no-tool/no-reminder/no-upload/no-monitoring/no-search/no-file
  /no-planning/no-control signals.
- tool text inferred features:
  agent/plugin, file, browser, search, message, calendar, reminder, monitoring,
  emotion, audio, camera, home, travel, code, research, study, social, external,
  proactive.
- interaction conflict/opportunity features:
  refusal-domain conflict, no-monitoring x monitoring tool, no-planning x travel,
  companion x proactive, agentic control opportunity.
- cross features:
  raw score and raw/policy interactions for linear-friendly modeling.

All features are soft continuous signals in `[-1, 1]`; none are hard blocks.

## Model Integration

For the first implementation, policy features are concatenated onto the existing
lexical side-channel:

```text
side_channel = lexical_features + policy_features
```

The first V3 config uses:

```text
LateInteractionRegressor
lexical_fusion = gated_add
head = linear
loss = SmoothL1 + LambdaNDCG@5
```

This makes the added model cost tiny: one extra linear projection/gate over a
small side-channel. The Jina sequence cache is reused as-is.

## Why This Fits P0

Current V2 mostly asks whether a plugin is semantically related to a
conversation. V3 adds soft signals for whether recommending or controlling a
plugin fits the user moment:

- user explicitly asked for action vs. wants companionship
- user refused a domain or capability
- plugin text implies monitoring, messaging, browser, file, or external action
- semantic relevance conflicts with a boundary

This directly targets observed bad cases where a tool is topically relevant but
should not be recommended or controlled.
