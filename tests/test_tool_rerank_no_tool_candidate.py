import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "tool_rerank_baseline"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from data import (  # noqa: E402
    ASK_CONFIRM_ID,
    NO_TOOL_ID,
    ConversationGroup,
    PairExample,
    add_ask_confirm_candidates,
    add_no_tool_candidates,
)
from metrics import compute_metrics  # noqa: E402


def _example(
    *,
    conversation_id: str,
    tool_id: str,
    label: float,
    scenario_type: str,
    relevance_mode: str,
) -> PairExample:
    return PairExample(
        sample_id=f"{conversation_id}_{tool_id}",
        conversation_id=conversation_id,
        tool_id=tool_id,
        label=label,
        raw_score=0.0,
        conversation_text="user: 先别调用工具，只陪我想想。",
        tool_text=f"tool_id: {tool_id}\nkind: plugin\n用于执行外部动作。",
        tool_fields={},
        scenario_type=scenario_type,
        relevance_mode=relevance_mode,
    )


def test_add_no_tool_candidate_labels_inactive_and_active_groups() -> None:
    groups = [
        ConversationGroup(
            conversation_id="conv_no_tool",
            examples=[
                _example(
                    conversation_id="conv_no_tool",
                    tool_id="plugin.monitor",
                    label=1.2,
                    scenario_type="boundary_or_refusal_no_tool",
                    relevance_mode="no_tool_or_low_relevance",
                )
            ],
        ),
        ConversationGroup(
            conversation_id="conv_active",
            examples=[
                _example(
                    conversation_id="conv_active",
                    tool_id="agent.browser_use",
                    label=2.1,
                    scenario_type="actionable_browser_task",
                    relevance_mode="high_relevance",
                )
            ],
        ),
    ]

    augmented = add_no_tool_candidates(groups, no_tool_label=2.5, active_label=-0.5, activation_threshold=1.0)
    by_id = {
        group.conversation_id: {example.tool_id: example for example in group.examples}
        for group in augmented
    }

    assert by_id["conv_no_tool"][NO_TOOL_ID].label == 2.5
    assert by_id["conv_active"][NO_TOOL_ID].label == -0.5
    assert by_id["conv_no_tool"][NO_TOOL_ID].conversation_text == groups[0].examples[0].conversation_text


def test_no_tool_candidate_label_is_above_existing_no_tool_group_max() -> None:
    group = ConversationGroup(
        conversation_id="conv_no_tool",
        examples=[
            _example(
                conversation_id="conv_no_tool",
                tool_id="plugin.music_practice_coach",
                label=3.1,
                scenario_type="companion_chat_no_tool",
                relevance_mode="no_tool_or_low_relevance",
            )
        ],
    )

    augmented = add_no_tool_candidates([group], no_tool_label=2.5, active_label=-0.5, activation_threshold=1.0)
    no_tool = next(example for example in augmented[0].examples if example.tool_id == NO_TOOL_ID)

    assert no_tool.label == 3.35


def test_add_ask_confirm_candidate_labels_weak_authorization_groups() -> None:
    groups = [
        ConversationGroup(
            conversation_id="conv_weak",
            examples=[
                _example(
                    conversation_id="conv_weak",
                    tool_id="plugin.coupon_finder",
                    label=1.7,
                    scenario_type="proactive_context_weak_tool",
                    relevance_mode="weak_or_requires_confirmation",
                )
            ],
        ),
        ConversationGroup(
            conversation_id="conv_active",
            examples=[
                _example(
                    conversation_id="conv_active",
                    tool_id="agent.browser_use",
                    label=2.1,
                    scenario_type="explicit_plugin_action",
                    relevance_mode="actionable_tool_relevance",
                )
            ],
        ),
    ]

    augmented = add_ask_confirm_candidates(groups, ask_confirm_label=2.5, inactive_label=-0.5)
    by_id = {
        group.conversation_id: {example.tool_id: example for example in group.examples}
        for group in augmented
    }

    assert by_id["conv_weak"][ASK_CONFIRM_ID].label == 2.5
    assert by_id["conv_active"][ASK_CONFIRM_ID].label == -0.5


def test_ask_confirm_candidate_can_cap_real_tool_labels() -> None:
    group = ConversationGroup(
        conversation_id="conv_weak",
        examples=[
            _example(
                conversation_id="conv_weak",
                tool_id="plugin.tempting_action",
                label=3.0,
                scenario_type="screen_context_weak_tool",
                relevance_mode="weak_or_requires_confirmation",
            )
        ],
    )

    augmented = add_ask_confirm_candidates(
        [group],
        ask_confirm_label=2.5,
        inactive_label=-0.5,
        label_margin=0.5,
        real_tool_ceiling_margin=0.75,
    )
    by_id = {example.tool_id: example for example in augmented[0].examples}

    assert by_id[ASK_CONFIRM_ID].label == 3.5
    assert by_id["plugin.tempting_action"].label == 2.75


def test_activation_metrics_with_no_tool_candidate() -> None:
    labels = np.asarray([2.5, 1.0, 2.0, -0.5], dtype="float32")
    predictions = np.asarray([2.2, 1.4, 2.1, 0.2], dtype="float32")
    conversation_ids = ["conv_no_tool", "conv_no_tool", "conv_active", "conv_active"]
    tool_ids = [NO_TOOL_ID, "plugin.monitor", "agent.browser_use", NO_TOOL_ID]

    metrics = compute_metrics(
        labels=labels,
        predictions=predictions,
        conversation_ids=conversation_ids,
        tool_ids=tool_ids,
        high_label_threshold=1.0,
        high_pred_threshold=1.0,
    )

    assert metrics["activation_accuracy"] == 1.0
    assert metrics["no_tool_recall"] == 1.0
    assert metrics["no_tool_precision"] == 1.0
    assert metrics["active_tool_top1"] == 1.0
    assert metrics["bad_exposed_top3_rate"] == 0.0
    assert metrics["bad_exposed_top5_rate"] == 0.0
    assert metrics["activation_action_accuracy"] == 1.0


def test_no_tool_top1_suppresses_bad_exposure_metrics() -> None:
    labels = np.asarray([2.5, -1.2, 1.1, -1.3, 2.0, -1.4], dtype="float32")
    predictions = np.asarray([2.4, 2.1, 1.8, 1.7, 2.2, 1.9], dtype="float32")
    conversation_ids = [
        "conv_no_tool",
        "conv_no_tool",
        "conv_no_tool",
        "conv_no_tool",
        "conv_active",
        "conv_active",
    ]
    tool_ids = [
        NO_TOOL_ID,
        "plugin.bad_monitor",
        "plugin.ok_reader",
        "plugin.bad_reply",
        "agent.browser_use",
        "plugin.bad_uploader",
    ]

    metrics = compute_metrics(
        labels=labels,
        predictions=predictions,
        conversation_ids=conversation_ids,
        tool_ids=tool_ids,
        high_label_threshold=1.0,
        high_pred_threshold=1.0,
    )

    assert metrics["bad_top3_rate"] == 1.0
    assert metrics["bad_exposed_top3_rate"] == 0.5
    assert metrics["bad_exposed_top5_rate"] == 0.5
    assert metrics["no_tool_suppressed_bad_top3_rate"] == 1.0
    assert metrics["active_bad_top3_rate"] == 1.0


def test_ask_confirm_action_metrics_and_suppressed_exposure() -> None:
    labels = np.asarray([2.5, 1.7, -1.2, 2.0, -0.5], dtype="float32")
    predictions = np.asarray([2.4, 2.1, 1.9, 2.2, 0.2], dtype="float32")
    conversation_ids = [
        "conv_confirm",
        "conv_confirm",
        "conv_confirm",
        "conv_active",
        "conv_active",
    ]
    tool_ids = [
        ASK_CONFIRM_ID,
        "plugin.weak_action",
        "plugin.bad_action",
        "agent.browser_use",
        ASK_CONFIRM_ID,
    ]

    metrics = compute_metrics(
        labels=labels,
        predictions=predictions,
        conversation_ids=conversation_ids,
        tool_ids=tool_ids,
        high_label_threshold=1.0,
        high_pred_threshold=1.0,
    )

    assert metrics["activation_action_accuracy"] == 1.0
    assert metrics["ask_confirm_recall"] == 1.0
    assert metrics["ask_confirm_precision"] == 1.0
    assert metrics["bad_top3_rate"] == 0.5
    assert metrics["bad_exposed_top3_rate"] == 0.0
