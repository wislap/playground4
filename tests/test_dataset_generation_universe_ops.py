from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse
from tool_relevance_lab.dataset_generation.universe_ops import (
    merge_tool_universes,
    validate_synthetic_universe,
)


def test_validate_synthetic_universe_rejects_runtime_state() -> None:
    real = ToolUniverse(
        tool_universe_id="real",
        tools=[ToolRecord(tool_id="plugin.web_search", kind="plugin", source_text="real")],
    )
    synthetic = ToolUniverse(
        tool_universe_id="synthetic",
        tools=[
            ToolRecord(
                tool_id="plugin.synthetic_bad",
                kind="plugin",
                source_text="kind: plugin\nstatus: running",
                metadata={"id": "synthetic_bad", "runtime": {"enabled": True}},
            ),
            ToolRecord(tool_id="plugin.web_search", kind="plugin", source_text="conflict"),
        ],
    )

    report = validate_synthetic_universe(synthetic, real_universe=real)

    assert not report.ok
    assert report.conflicting_tool_ids == ["plugin.web_search"]
    assert report.forbidden_metadata_tool_ids == ["plugin.synthetic_bad"]
    assert report.forbidden_source_text_tool_ids == ["plugin.synthetic_bad"]


def test_merge_tool_universes_writes_combined_universe(tmp_path) -> None:
    real = ToolUniverse(
        tool_universe_id="real",
        tools=[ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser")],
    )
    synthetic = ToolUniverse(
        tool_universe_id="synthetic",
        tools=[ToolRecord(tool_id="plugin.synthetic_demo", kind="plugin", source_text="demo")],
    )

    merged = merge_tool_universes(
        output_dir=tmp_path / "mixed",
        output_universe_id="mixed_v1",
        universes=[real, synthetic],
    )

    assert merged.tool_universe_id == "mixed_v1"
    assert [tool.tool_id for tool in merged.tools] == [
        "agent.browser_use",
        "plugin.synthetic_demo",
    ]
    assert (tmp_path / "mixed" / "manifest.json").exists()
    assert (tmp_path / "mixed" / "tools.jsonl").exists()
