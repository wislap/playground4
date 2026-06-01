import sys
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parents[3] / "modules" / "v5_action_policy" / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from v5_action_policy import DesiredStateGapPolicy, ToolState, action_feedback_advantages  # noqa: E402


def test_v5_recommends_enable_and_disable_from_state_gap() -> None:
    policy = DesiredStateGapPolicy(
        mu=0.0,
        temperature=1.0,
        enable_threshold=0.75,
        disable_threshold=0.75,
        action_temperature=0.10,
    )
    tools = [
        ToolState(tool_id="plugin.good_disabled", score=2.0, enabled=False),
        ToolState(tool_id="plugin.bad_enabled", score=-2.0, enabled=True),
        ToolState(tool_id="plugin.ok_disabled", score=0.2, enabled=False),
    ]

    actions = policy.recommend(tools)

    assert {(action.tool_id, action.action) for action in actions} == {
        ("plugin.good_disabled", "enable"),
        ("plugin.bad_enabled", "disable"),
    }
    assert len(actions) == 2
    assert all(action.action_probability > 0.5 for action in actions)


def test_v5_empty_actions_when_state_is_already_close_to_desired() -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.90, disable_threshold=0.90)
    tools = [
        ToolState(tool_id="plugin.mid_disabled", score=0.0, enabled=False),
        ToolState(tool_id="plugin.mid_enabled", score=0.0, enabled=True),
    ]

    assert policy.recommend(tools) == []


def test_v5_max_actions_caps_without_fixed_topk_requirement() -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.65, disable_threshold=0.65)
    tools = [
        ToolState(tool_id="plugin.a", score=3.0, enabled=False),
        ToolState(tool_id="plugin.b", score=2.0, enabled=False),
        ToolState(tool_id="plugin.c", score=-3.0, enabled=True),
    ]

    all_actions = policy.recommend(tools)
    capped_actions = policy.recommend(tools, max_actions=2)

    assert len(all_actions) == 3
    assert len(capped_actions) == 2
    assert [action.tool_id for action in capped_actions] == [action.tool_id for action in all_actions[:2]]


def test_v5_log_prob_has_gradients_for_policy_parameters_and_scores() -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.75, disable_threshold=0.75)
    scores = torch.tensor([2.0, -2.0], dtype=torch.float32, requires_grad=True)
    enabled = torch.tensor([False, True])
    selected = torch.tensor([True, True])

    loss = -policy.log_prob(scores=scores, enabled=enabled, selected=selected).sum()
    loss.backward()

    assert scores.grad is not None
    assert scores.grad[0] < 0.0
    assert scores.grad[1] > 0.0
    assert policy.mu.grad is not None
    assert policy.enable_threshold_logit.grad is not None
    assert policy.disable_threshold_logit.grad is not None


def test_v5_action_feedback_advantages_map_to_v22_score_direction() -> None:
    policy = DesiredStateGapPolicy(mu=0.0, temperature=1.0, enable_threshold=0.65, disable_threshold=0.65)
    actions = policy.recommend(
        [
            ToolState(tool_id="plugin.enable_good", score=2.0, enabled=False),
            ToolState(tool_id="plugin.disable_good", score=-2.0, enabled=True),
            ToolState(tool_id="plugin.enable_bad", score=1.8, enabled=False),
        ]
    )

    by_id = {action.tool_id: action for action in actions}
    advantages = action_feedback_advantages(
        [by_id["plugin.enable_good"], by_id["plugin.disable_good"], by_id["plugin.enable_bad"]],
        [1.0, 0.5, -1.0],
    )

    assert advantages["plugin.enable_good"] > 0.0
    assert advantages["plugin.disable_good"] < 0.0
    assert advantages["plugin.enable_bad"] < 0.0
