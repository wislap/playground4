from __future__ import annotations

import json
from pathlib import Path

from tool_relevance_lab.dataset_generation.jsonl import read_model_jsonl, write_jsonl
from tool_relevance_lab.dataset_generation.schemas import (
    ToolRecord,
    ToolUniverse,
    ToolUniverseManifest,
)


def write_tool_universe_dir(path: Path, universe: ToolUniverse, manifest: ToolUniverseManifest) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if manifest.tool_universe_id != universe.tool_universe_id:
        raise ValueError("manifest and universe ids must match")
    if manifest.tool_count != len(universe.tools):
        raise ValueError("manifest tool_count must match universe tools")
    (path / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(path / "tools.jsonl", universe.tools)


def read_tool_universe(path: Path) -> ToolUniverse:
    if path.is_dir():
        manifest_path = path / "manifest.json"
        tools_path = path / "tools.jsonl"
        manifest = ToolUniverseManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        tools = read_model_jsonl(tools_path, ToolRecord)
        if manifest.tool_count != len(tools):
            raise ValueError("manifest tool_count does not match tools.jsonl")
        return ToolUniverse(
            tool_universe_id=manifest.tool_universe_id,
            created_at=manifest.created_at,
            tools=tools,
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    return ToolUniverse.model_validate(raw)


def load_tool_universes(paths: list[Path], *, tool_universe_id: str | None = None) -> ToolUniverse:
    """Load one or more tool universes into a single in-memory view."""
    if not paths:
        raise ValueError("at least one tool universe path is required")
    universes = [read_tool_universe(path) for path in paths]
    if len(universes) == 1 and tool_universe_id is None:
        return universes[0]

    tools: list[ToolRecord] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for universe in universes:
        for tool in universe.tools:
            if tool.tool_id in seen:
                duplicates.append(tool.tool_id)
                continue
            seen.add(tool.tool_id)
            tools.append(tool)
    if duplicates:
        raise ValueError(f"duplicate tool ids while loading universes: {sorted(set(duplicates))}")

    universe_id = tool_universe_id or "+".join(universe.tool_universe_id for universe in universes)
    created_at = universes[0].created_at if len(universes) == 1 else None
    return ToolUniverse(tool_universe_id=universe_id, created_at=created_at, tools=tools)
