import pytest

from tool_relevance_lab.dataset_generation.synthetic_plugins import (
    SyntheticPluginSpec,
    synthetic_plugin_to_tool,
)


def test_synthetic_plugin_rejects_runtime_fields() -> None:
    with pytest.raises(ValueError, match="runtime"):
        SyntheticPluginSpec.model_validate(
            {
                "id": "daily_briefing",
                "name": "每日简报",
                "type": "plugin",
                "description": "生成每日简报。",
                "short_description": "Generate a daily briefing.",
                "keywords": ["每日简报", "daily briefing"],
                "passive": False,
                "runtime": {"enabled": True},
            }
        )


def test_synthetic_plugin_normalizes_tool_id_prefix() -> None:
    spec = SyntheticPluginSpec.model_validate(
        {
            "id": "plugin.daily_briefing",
            "name": "每日简报",
            "type": "plugin",
            "description": "生成每日简报。",
            "short_description": "Generate a daily briefing.",
            "keywords": ["每日简报", "daily briefing"],
            "passive": False,
        }
    )

    assert spec.id == "daily_briefing"


def test_synthetic_plugin_tool_source_text_is_static_only() -> None:
    spec = SyntheticPluginSpec(
        id="daily_briefing",
        name="每日简报",
        type="plugin",
        description="汇总用户一天开始前可能关心的信息。",
        short_description="Generate a concise daily briefing.",
        keywords=["每日简报", "daily briefing", "morning"],
        passive=False,
    )

    tool = synthetic_plugin_to_tool(spec)

    assert tool.tool_id == "plugin.daily_briefing"
    assert "runtime" not in tool.source_text
    assert "status" not in tool.source_text
    assert "available" not in tool.model_dump()
