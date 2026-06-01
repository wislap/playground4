import asyncio
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "tool_rerank_baseline"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from v5_action_policy import DesiredStateGapPolicy, ToolState  # noqa: E402
from v5_grpo import compute_v5_multiscale_grpo_loss  # noqa: E402
from v5_rollout import (  # noqa: E402
    V5RolloutState,
    build_v5_judge_prompt,
    judge_v5_rollout_states,
    parse_v5_judge_response,
    sample_v5_slates,
)
from tool_relevance_lab.dataset_generation.jsonl import read_jsonl  # noqa: E402


def _state() -> V5RolloutState:
    return V5RolloutState(
        state_id="state_001",
        conversation_excerpt="用户说先别打扰太多，但日历提醒可能有用。",
        tools=[
            ToolState(tool_id="plugin.calendar", score=2.0, enabled=False),
            ToolState(tool_id="plugin.browser", score=0.1, enabled=False),
            ToolState(tool_id="plugin.monitor", score=-2.0, enabled=True),
            ToolState(tool_id="plugin.music", score=0.0, enabled=True),
        ],
    )


def test_sample_v5_slates_builds_multiscale_group() -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.70, disable_threshold=0.70)
    slates = sample_v5_slates(policy=policy, state=_state(), group_size=5, max_actions_per_slate=3, seed=7)

    assert len(slates) == 5
    assert [slate.kind for slate in slates] == [
        "anchor",
        "conservative",
        "sampled",
        "over_recommend",
        "hard_negative",
    ]
    assert all(len(slate.actions) <= 3 for slate in slates)
    assert any(slate.actions for slate in slates)
    assert all(len(slate.selected_mask) == len(_state().tools) for slate in slates)


def test_v5_judge_prompt_and_response_parser_round_trip() -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.70, disable_threshold=0.70)
    state = _state()
    slates = sample_v5_slates(policy=policy, state=state, group_size=5, seed=3)

    prompt = build_v5_judge_prompt(state=state, slates=slates)

    assert prompt[0]["role"] == "system"
    assert "candidate_slates" in prompt[1]["content"]

    payload = {
        "state_id": state.state_id,
        "slates": [
            {
                "slate_id": slate.slate_id,
                "slate_score": 1.0 if slate.kind == "anchor" else 0.0,
                "action_scores": [
                    {"tool_id": action.tool_id, "action": action.action, "score": 0.5}
                    for action in slate.actions
                ],
            }
            for slate in slates
        ],
        "rationale": "anchor is least disruptive.",
    }
    record = parse_v5_judge_response(
        json.dumps(payload),
        state_id=state.state_id,
        expected_slate_ids={slate.slate_id for slate in slates},
    )

    assert record.state_id == state.state_id
    assert len(record.judged_slates) == len(slates)
    assert record.rationale == "anchor is least disruptive."


def test_v5_multiscale_grpo_loss_updates_policy_parameters() -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.70, disable_threshold=0.70)
    state = _state()
    slates = sample_v5_slates(policy=policy, state=state, group_size=5, seed=11)
    payload = {
        "state_id": state.state_id,
        "slates": [
            {
                "slate_id": slate.slate_id,
                "slate_score": 1.0 if slate.kind == "anchor" else -0.5,
                "action_scores": [
                    {
                        "tool_id": action.tool_id,
                        "action": action.action,
                        "score": 0.9 if action.tool_id in {"plugin.calendar", "plugin.monitor"} else -0.7,
                    }
                    for action in slate.actions
                ],
            }
            for slate in slates
        ],
        "rationale": "calendar enable and monitor disable are preferred.",
    }
    judge = parse_v5_judge_response(
        json.dumps(payload),
        state_id=state.state_id,
        expected_slate_ids={slate.slate_id for slate in slates},
    )

    group_loss = compute_v5_multiscale_grpo_loss(
        policy=policy,
        tools=state.tools,
        slates=slates,
        judge=judge,
    )
    group_loss.loss.backward()

    assert torch.isfinite(group_loss.loss)
    assert policy.mu.grad is not None
    assert policy.enable_threshold_logit.grad is not None
    assert policy.disable_threshold_logit.grad is not None


def test_judge_v5_rollout_states_dry_run_writes_resumable_jsonl(tmp_path: Path) -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.70, disable_threshold=0.70)
    output_path = tmp_path / "v5_preferences.jsonl"
    error_path = tmp_path / "errors.jsonl"

    summary = asyncio.run(
        judge_v5_rollout_states(
            policy=policy,
            states=[_state()],
            client=None,
            output_path=output_path,
            error_path=error_path,
            model="dry-run-judge",
            concurrency=2,
            dry_run=True,
        )
    )

    rows = list(read_jsonl(output_path))
    assert summary.succeeded == 1
    assert rows[0]["state"]["state_id"] == "state_001"
    assert len(rows[0]["slates"]) == 5
    assert len(rows[0]["judge"]["slates"]) == 5
