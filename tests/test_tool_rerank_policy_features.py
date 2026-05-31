import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "tool_rerank_baseline"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from data import PairExample  # noqa: E402
from policy_features import ACTION_CONSENT_B_FEATURES, ACTION_CONSENT_FEATURES, PolicyFeatureBuilder  # noqa: E402


def _example(*, conversation: str, tool_id: str, tool_text: str) -> PairExample:
    return PairExample(
        sample_id="sample_1",
        conversation_id="conv_1",
        tool_id=tool_id,
        label=0.0,
        raw_score=0.0,
        conversation_text=conversation,
        tool_text=tool_text,
        tool_fields={},
        scenario_type="boundary_or_refusal_no_tool",
        relevance_mode="no_tool_or_low_relevance",
    )


def test_action_consent_features_capture_monitoring_refusal_conflict() -> None:
    builder = PolicyFeatureBuilder(groups=("action_consent",))
    example = _example(
        conversation="用户说：先别监听我的情绪，也别提醒我开安抚灯。今天只想安静写论文。",
        tool_id="plugin.emotion_lamp_listener",
        tool_text="监听本地语音语调和对话情绪变化，将房间氛围灯调整为安抚模式。",
    )

    row = builder.transform([example])[0]
    values = dict(zip(builder.feature_names, row))

    assert builder.feature_names == ACTION_CONSENT_FEATURES
    assert row.dtype == np.float32
    assert values["ac_no_call_intent"] > 0.0
    assert values["ac_denies_monitoring"] > 0.0
    assert values["ac_tool_continuous_monitoring"] > 0.0
    assert values["ac_conflict_denied_monitoring"] > 0.0
    assert values["ac_no_call_side_effect_risk"] > 0.0


def test_action_consent_features_capture_browser_form_fill_match() -> None:
    builder = PolicyFeatureBuilder(groups=("action_consent",))
    example = _example(
        conversation=(
            "帮我打开这个网页链接，填写志愿者报名表。"
            "最后到提交前停一下给我确认，别直接提交。"
        ),
        tool_id="agent.browser_use",
        tool_text="本地浏览器自动化，适合打开 URL、填写网页表单、网页搜索、从网络下载。",
    )

    row = builder.transform([example])[0]
    values = dict(zip(builder.feature_names, row))

    assert values["ac_can_call_intent"] > 0.0
    assert values["ac_requested_browser"] > 0.0
    assert values["ac_requested_form_fill"] > 0.0
    assert values["ac_match_requested_browser"] > 0.0
    assert values["ac_match_requested_form_fill"] > 0.0
    assert values["ac_requires_final_confirmation"] > 0.0
    assert values["ac_conflict_final_submit"] > 0.0


def test_action_consent_b_features_capture_monitoring_refusal_veto() -> None:
    builder = PolicyFeatureBuilder(groups=("action_consent_b",))
    example = _example(
        conversation="用户说：先别监听我的情绪，也别提醒我开安抚灯。今天只想安静写论文。",
        tool_id="plugin.emotion_lamp_listener",
        tool_text="监听本地语音语调和对话情绪变化，将房间氛围灯调整为安抚模式。",
    )

    row = builder.transform([example])[0]
    values = dict(zip(builder.feature_names, row))

    assert builder.feature_names == ACTION_CONSENT_B_FEATURES
    assert len(builder.feature_names) < len(ACTION_CONSENT_FEATURES)
    assert row.dtype == np.float32
    assert values["acb_no_call_intent"] > 0.0
    assert values["acb_tool_background_monitor"] > 0.0
    assert values["acb_tool_user_state_inference"] > 0.0
    assert values["acb_denied_operation_conflict"] > 0.0
    assert values["acb_no_call_side_effect_conflict"] > 0.0
    assert values["acb_refusal_background_monitor_veto"] > 0.0
    assert values["acb_refusal_user_state_veto"] > 0.0


def test_action_consent_b_features_keep_browser_match_but_avoid_device_strong_match() -> None:
    builder = PolicyFeatureBuilder(groups=("action_consent_b",))
    browser_example = _example(
        conversation=(
            "帮我打开这个网页链接，填写志愿者报名表。"
            "最后到提交前停一下给我确认，别直接提交。"
        ),
        tool_id="agent.browser_use",
        tool_text="本地浏览器自动化，适合打开 URL、填写网页表单、网页搜索、从网络下载。",
    )
    device_example = _example(
        conversation="我在想睡前仪式里要不要用薰衣草和暖光，但先别控制设备，只陪我想想。",
        tool_id="plugin.aroma_ritual_mixer",
        tool_text="连接本地智能香薰机与灯带，根据聊天主题调配香型、扩香强度与氛围灯颜色。",
    )

    browser_values = dict(zip(builder.feature_names, builder.transform([browser_example])[0]))
    device_values = dict(zip(builder.feature_names, builder.transform([device_example])[0]))

    assert browser_values["acb_can_call_intent"] > 0.0
    assert browser_values["acb_weak_operation_match"] > 0.0
    assert browser_values["acb_strong_operation_match"] > 0.0
    assert browser_values["acb_confirmation_boundary_conflict"] > 0.0

    assert device_values["acb_tool_device_or_environment_control"] > 0.0
    assert device_values["acb_refusal_device_control_veto"] > 0.0
    assert device_values["acb_strong_operation_match"] == 0.0
