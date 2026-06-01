from tool_relevance_lab.dataset_generation.schemas import (
    ToolRecord,
    ToolUniverse,
    ToolUniverseManifest,
)
import pytest

from tool_relevance_lab.dataset_generation.tool_store import (
    load_tool_universes,
    read_tool_universe,
    write_tool_universe_dir,
)


def test_tool_universe_directory_round_trip(tmp_path) -> None:
    universe = ToolUniverse(
        tool_universe_id="mixed_v1",
        created_at="2026-05-26T00:00:00Z",
        tools=[
            ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
            ToolRecord(tool_id="plugin.synthetic_demo", kind="plugin", source_text="demo"),
        ],
    )
    manifest = ToolUniverseManifest(
        tool_universe_id="mixed_v1",
        created_at=universe.created_at,
        kind="mixed",
        tool_count=2,
        source={"note": "test"},
    )

    path = tmp_path / "mixed_v1"
    write_tool_universe_dir(path, universe, manifest)
    loaded = read_tool_universe(path)

    assert loaded.tool_universe_id == universe.tool_universe_id
    assert [tool.tool_id for tool in loaded.tools] == [
        "agent.browser_use",
        "plugin.synthetic_demo",
    ]


def test_load_tool_universes_combines_multiple_sources(tmp_path) -> None:
    real = ToolUniverse(
        tool_universe_id="real_v1",
        tools=[ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser")],
    )
    synthetic = ToolUniverse(
        tool_universe_id="synthetic_v1",
        tools=[ToolRecord(tool_id="plugin.synthetic_demo", kind="plugin", source_text="demo")],
    )
    write_tool_universe_dir(
        tmp_path / "real",
        real,
        ToolUniverseManifest(tool_universe_id="real_v1", tool_count=1),
    )
    write_tool_universe_dir(
        tmp_path / "synthetic",
        synthetic,
        ToolUniverseManifest(tool_universe_id="synthetic_v1", tool_count=1),
    )

    loaded = load_tool_universes(
        [tmp_path / "real", tmp_path / "synthetic"],
        tool_universe_id="runtime_pool_v1",
    )

    assert loaded.tool_universe_id == "runtime_pool_v1"
    assert [tool.tool_id for tool in loaded.tools] == [
        "agent.browser_use",
        "plugin.synthetic_demo",
    ]


def test_load_tool_universes_rejects_duplicate_ids(tmp_path) -> None:
    first = ToolUniverse(
        tool_universe_id="first",
        tools=[ToolRecord(tool_id="plugin.same", kind="plugin", source_text="one")],
    )
    second = ToolUniverse(
        tool_universe_id="second",
        tools=[ToolRecord(tool_id="plugin.same", kind="plugin", source_text="two")],
    )
    write_tool_universe_dir(
        tmp_path / "first",
        first,
        ToolUniverseManifest(tool_universe_id="first", tool_count=1),
    )
    write_tool_universe_dir(
        tmp_path / "second",
        second,
        ToolUniverseManifest(tool_universe_id="second", tool_count=1),
    )

    with pytest.raises(ValueError, match="duplicate tool ids"):
        load_tool_universes([tmp_path / "first", tmp_path / "second"])
