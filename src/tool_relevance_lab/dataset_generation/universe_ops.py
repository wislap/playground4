from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse, ToolUniverseManifest
from tool_relevance_lab.dataset_generation.tool_store import write_tool_universe_dir


FORBIDDEN_SYNTHETIC_TEXT = (
    "runtime.enabled",
    "runtime.auto_start",
    "status:",
    "available:",
    "source_path",
)
FORBIDDEN_SYNTHETIC_METADATA = {
    "runtime",
    "plugin_runtime",
    "status",
    "available",
    "enabled",
    "auto_start",
    "source_path",
}


@dataclass(frozen=True)
class SyntheticUniverseValidationReport:
    tool_count: int
    duplicate_tool_ids: list[str]
    conflicting_tool_ids: list[str]
    non_plugin_tool_ids: list[str]
    forbidden_metadata_tool_ids: list[str]
    forbidden_source_text_tool_ids: list[str]

    @property
    def ok(self) -> bool:
        return not any(
            [
                self.duplicate_tool_ids,
                self.conflicting_tool_ids,
                self.non_plugin_tool_ids,
                self.forbidden_metadata_tool_ids,
                self.forbidden_source_text_tool_ids,
            ]
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def validate_synthetic_universe(
    synthetic_universe: ToolUniverse,
    *,
    real_universe: ToolUniverse | None = None,
) -> SyntheticUniverseValidationReport:
    counts = Counter(tool.tool_id for tool in synthetic_universe.tools)
    duplicate_tool_ids = sorted(tool_id for tool_id, count in counts.items() if count > 1)
    real_ids = {tool.tool_id for tool in real_universe.tools} if real_universe else set()
    synthetic_ids = {tool.tool_id for tool in synthetic_universe.tools}
    conflicting_tool_ids = sorted(real_ids & synthetic_ids)

    non_plugin_tool_ids: list[str] = []
    forbidden_metadata_tool_ids: list[str] = []
    forbidden_source_text_tool_ids: list[str] = []
    for tool in synthetic_universe.tools:
        if tool.kind != "plugin":
            non_plugin_tool_ids.append(tool.tool_id)
        if FORBIDDEN_SYNTHETIC_METADATA & set(tool.metadata):
            forbidden_metadata_tool_ids.append(tool.tool_id)
        if any(token in tool.source_text for token in FORBIDDEN_SYNTHETIC_TEXT):
            forbidden_source_text_tool_ids.append(tool.tool_id)

    return SyntheticUniverseValidationReport(
        tool_count=len(synthetic_universe.tools),
        duplicate_tool_ids=duplicate_tool_ids,
        conflicting_tool_ids=conflicting_tool_ids,
        non_plugin_tool_ids=sorted(non_plugin_tool_ids),
        forbidden_metadata_tool_ids=sorted(forbidden_metadata_tool_ids),
        forbidden_source_text_tool_ids=sorted(forbidden_source_text_tool_ids),
    )


def merge_tool_universes(
    *,
    output_dir: Path,
    output_universe_id: str,
    universes: list[ToolUniverse],
    kind: str = "mixed",
) -> ToolUniverse:
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
        raise ValueError(f"duplicate tool ids while merging: {sorted(set(duplicates))}")

    merged = ToolUniverse(
        tool_universe_id=output_universe_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        tools=tools,
    )
    manifest = ToolUniverseManifest(
        tool_universe_id=output_universe_id,
        created_at=merged.created_at,
        kind=kind,
        tool_count=len(tools),
        source={
            "type": "merged",
            "input_universe_ids": [universe.tool_universe_id for universe in universes],
        },
    )
    write_tool_universe_dir(output_dir, merged, manifest)
    return merged
