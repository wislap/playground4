from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tool_relevance_lab.dataset_generation.tool_store import load_tool_universes  # noqa: E402


@dataclass(frozen=True)
class PairExample:
    sample_id: str
    conversation_id: str
    tool_id: str
    label: float
    raw_score: float
    conversation_text: str
    tool_text: str
    tool_fields: dict[str, str]
    scenario_type: str
    relevance_mode: str


@dataclass(frozen=True)
class ConversationGroup:
    conversation_id: str
    examples: list[PairExample]


@dataclass(frozen=True)
class DatasetBundle:
    train_groups: list[ConversationGroup]
    val_groups: list[ConversationGroup]
    train_examples: list[PairExample]
    val_examples: list[PairExample]


def load_dataset(config: dict[str, Any]) -> DatasetBundle:
    all_groups = load_all_groups(config)
    data_cfg = config["data"]
    train_groups, val_groups = split_groups(
        all_groups,
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        seed=int(data_cfg.get("seed", 20260526)),
    )
    return bundle_from_groups(train_groups=train_groups, val_groups=val_groups)


def load_all_groups(config: dict[str, Any]) -> list[ConversationGroup]:
    data_cfg = config["data"]
    run_dir = REPO_ROOT / data_cfg["run_dir"]
    tools = load_tool_universes(
        [REPO_ROOT / path for path in data_cfg["tool_universe_paths"]],
        tool_universe_id=data_cfg.get("tool_universe_id"),
    ).by_id()

    conversations = {
        row["conversation_id"]: row for row in _read_jsonl(run_dir / "conversations.jsonl")
    }
    candidate_sets = {
        row["sample_id"]: row for row in _read_jsonl(run_dir / "candidate_sets.jsonl")
    }

    label_field = data_cfg.get("label_field", "confidence")
    grouped: dict[str, list[PairExample]] = {}
    for row in _read_jsonl(run_dir / "calibrated.jsonl"):
        sample_id = row["sample_id"]
        tool_id = row["tool_id"]
        candidate = candidate_sets[sample_id]
        conversation_id = candidate["conversation_id"]
        conversation = conversations[conversation_id]
        tool = tools[tool_id]
        extra = conversation.get("provenance", {}).get("extra", {})
        example = PairExample(
            sample_id=sample_id,
            conversation_id=conversation_id,
            tool_id=tool_id,
            label=float(row[label_field]),
            raw_score=float(row.get("raw_score", 0.0)),
            conversation_text=render_conversation(conversation),
            tool_text=render_tool(tool.model_dump(mode="json")),
            tool_fields=render_tool_fields(tool.model_dump(mode="json")),
            scenario_type=str(extra.get("scenario_type", "unknown")),
            relevance_mode=str(extra.get("tool_relevance_mode", "unknown")),
        )
        grouped.setdefault(conversation_id, []).append(example)

    all_groups = [
        ConversationGroup(conversation_id=conversation_id, examples=examples)
        for conversation_id, examples in sorted(grouped.items())
    ]
    return all_groups


def bundle_from_groups(
    *,
    train_groups: list[ConversationGroup],
    val_groups: list[ConversationGroup],
) -> DatasetBundle:
    return DatasetBundle(
        train_groups=train_groups,
        val_groups=val_groups,
        train_examples=[example for group in train_groups for example in group.examples],
        val_examples=[example for group in val_groups for example in group.examples],
    )


def split_groups(
    groups: list[ConversationGroup],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[ConversationGroup], list[ConversationGroup]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_ids = {group.conversation_id for group in shuffled[:val_count]}
    train_groups = [group for group in groups if group.conversation_id not in val_ids]
    val_groups = [group for group in groups if group.conversation_id in val_ids]
    return train_groups, val_groups


def render_conversation(conversation: dict[str, Any]) -> str:
    lines = [f"trigger: {conversation.get('trigger', 'turn_end')}"]
    extra = conversation.get("provenance", {}).get("extra", {})
    if extra:
        lines.append(
            "context: "
            f"scenario={extra.get('scenario_type', 'unknown')}; "
            f"auth={extra.get('authorization_level', 'unknown')}; "
            f"actionability={extra.get('latest_user_actionability', 'unknown')}"
        )
    for message in conversation["messages"]:
        attachments = message.get("attachments") or []
        note = f" attachments={'; '.join(attachments)}" if attachments else ""
        lines.append(f"{message['role']}{note}: {message.get('text', '')}")
    return "\n".join(lines)


def render_tool(tool: dict[str, Any]) -> str:
    metadata = tool.get("metadata", {})
    name = metadata.get("name")
    prefix = f"tool_id: {tool['tool_id']}\nkind: {tool['kind']}"
    if name:
        prefix += f"\nname: {name}"
    return f"{prefix}\n{tool['source_text']}"


def render_tool_fields(tool: dict[str, Any]) -> dict[str, str]:
    metadata = tool.get("metadata", {})
    identity_parts = [
        f"tool_id: {tool['tool_id']}",
        f"kind: {tool['kind']}",
    ]
    for key in ("id", "name", "type"):
        if metadata.get(key):
            identity_parts.append(f"{key}: {metadata[key]}")

    description_parts = []
    for key in ("description", "short_description"):
        if metadata.get(key):
            description_parts.append(f"{key}: {metadata[key]}")
    if not description_parts:
        description_parts.append(str(tool["source_text"]))

    capability_parts = []
    for key in ("keywords", "capability_key", "method", "passive", "priority"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        capability_parts.append(f"{key}: {value}")

    examples_parts = []
    for key in ("examples", "trigger_examples", "use_cases"):
        value = metadata.get(key)
        if value:
            examples_parts.append(f"{key}: {value}")

    schema_parts = []
    for key in ("schema", "input_schema", "parameters"):
        value = metadata.get(key)
        if value:
            schema_parts.append(f"{key}: {value}")

    return {
        "identity": "\n".join(identity_parts),
        "description": "\n".join(description_parts),
        "capabilities": "\n".join(capability_parts) or "\n".join(description_parts),
        "examples": "\n".join(examples_parts) or "\n".join(description_parts),
        "schema": "\n".join(schema_parts) or "\n".join(identity_parts),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows
