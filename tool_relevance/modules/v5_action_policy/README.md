# V5 Action Policy

Canonical module for the V5 plugin state action policy.

V5 does not replace V2.2b semantic relevance. It consumes V2.2b scores and
current plugin enabled states, then recommends whether to enable or disable
plugins.

## Files

- `source/v5_action_policy.py`: desired-state gap policy and recommendation API.
- `source/v5_rollout_generation.py`: slate sampling, judge prompt, response parsing.
- `source/v5_grpo_training.py`: multi-scale GRPO loss for judged slates.

## Contract

- V2.2b owns relevance scoring.
- V5 owns action aggressiveness and state-change policy.
- V5 training data should live under `tool_relevance/runtime/outputs/` until promoted.
