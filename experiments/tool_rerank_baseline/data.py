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


NO_TOOL_ID = "__NO_TOOL__"
ASK_CONFIRM_ID = "__ASK_CONFIRM__"
NO_TOOL_SAMPLE_SUFFIX = "__no_tool__"
ASK_CONFIRM_SAMPLE_SUFFIX = "__ask_confirm__"
NO_TOOL_TEXT = "\n".join(
    [
        "tool_id: __NO_TOOL__",
        "kind: no_tool",
        "name: No tool invocation",
        (
            "Respond directly without invoking plugins, agents, browser automation, "
            "file access, monitoring, messaging, uploading, device control, or external actions."
        ),
        (
            "Use when the user is venting, asking for quiet companionship, refusing action, "
            "deferring confirmation, or when no candidate tool should be activated."
        ),
    ]
)
ASK_CONFIRM_TEXT = "\n".join(
    [
        "tool_id: __ASK_CONFIRM__",
        "kind: ask_confirm",
        "name: Ask for confirmation before invoking tools",
        (
            "Do not invoke a plugin or agent yet. Ask the user for explicit confirmation, "
            "missing details, authorization, or final approval before taking action."
        ),
        (
            "Use when the user intent is weak, ambiguous, assistant-suggested, lacks authorization, "
            "or requires confirmation before browsing, messaging, uploading, changing files, "
            "controlling devices, or performing external actions."
        ),
        (
            "This is the correct next action when the assistant should pause and ask permission "
            "instead of choosing a concrete tool."
        ),
    ]
)
NO_TOOL_FIELDS = {
    "identity": "tool_id: __NO_TOOL__\nkind: no_tool\nname: No tool invocation",
    "description": (
        "Respond directly without invoking tools. Do not monitor, control devices, upload, "
        "message, browse, edit files, or perform external actions."
    ),
    "capabilities": "direct response, quiet presence, clarification, no external action",
    "examples": (
        "The user asks not to use tools; the user only wants to talk; "
        "the user has not confirmed an external action."
    ),
    "schema": "no parameters",
}
ASK_CONFIRM_FIELDS = {
    "identity": "tool_id: __ASK_CONFIRM__\nkind: ask_confirm\nname: Ask confirmation",
    "description": (
        "Ask the user for explicit confirmation or missing details before invoking tools. "
        "Do not perform external actions yet."
    ),
    "capabilities": "confirmation, clarification, authorization request, final approval",
    "examples": (
        "The assistant suggested a tool but the user has not authorized it; "
        "the user asks to stop before submitting; the request is weak or ambiguous."
    ),
    "schema": "no parameters",
}


FACTOR_AXIS_NAMES: tuple[str, ...] = (
    "capability_match",
    "action_demand",
    "target_specificity",
    "consent_boundary",
    "intervention_cost",
    "companionship_fit",
)


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
    authorization_level: str = "unknown"
    latest_user_actionability: str = "unknown"
    latest_user_text: str = ""
    axis_labels: tuple[float, ...] | None = None


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
    require_axis_scores = bool(data_cfg.get("require_axis_scores", False))
    grouped: dict[str, list[PairExample]] = {}
    for row in _read_jsonl(run_dir / "calibrated.jsonl"):
        sample_id = row["sample_id"]
        tool_id = row["tool_id"]
        candidate = candidate_sets[sample_id]
        conversation_id = candidate["conversation_id"]
        conversation = conversations[conversation_id]
        tool = tools[tool_id]
        extra = conversation.get("provenance", {}).get("extra", {})
        axis_labels = _axis_labels_from_row(row)
        if require_axis_scores and axis_labels is None:
            raise ValueError(
                f"axis_scores are required but missing for sample_id={sample_id} tool_id={tool_id}"
            )
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
            authorization_level=str(extra.get("authorization_level", "unknown")),
            latest_user_actionability=str(extra.get("latest_user_actionability", "unknown")),
            latest_user_text=latest_user_text(conversation),
            axis_labels=axis_labels,
        )
        grouped.setdefault(conversation_id, []).append(example)

    all_groups = [
        ConversationGroup(conversation_id=conversation_id, examples=examples)
        for conversation_id, examples in sorted(grouped.items())
    ]
    no_tool_cfg = config.get("no_tool_candidate", {})
    if no_tool_cfg.get("enabled", False):
        all_groups = add_no_tool_candidates(
            all_groups,
            no_tool_label=float(no_tool_cfg.get("no_tool_label", 2.5)),
            active_label=float(no_tool_cfg.get("active_label", -0.5)),
            label_margin=float(no_tool_cfg.get("label_margin", 0.25)),
            activation_threshold=float(no_tool_cfg.get("activation_threshold", 1.0)),
            scenario_patterns=tuple(no_tool_cfg.get("scenario_patterns", DEFAULT_NO_TOOL_SCENARIO_PATTERNS)),
            relevance_patterns=tuple(no_tool_cfg.get("relevance_patterns", DEFAULT_NO_TOOL_RELEVANCE_PATTERNS)),
            latest_user_text_patterns=tuple(
                no_tool_cfg.get("latest_user_text_patterns", DEFAULT_NO_TOOL_LATEST_USER_TEXT_PATTERNS)
            ),
        )
    ask_confirm_cfg = config.get("ask_confirm_candidate", {})
    if ask_confirm_cfg.get("enabled", False):
        all_groups = add_ask_confirm_candidates(
            all_groups,
            ask_confirm_label=float(ask_confirm_cfg.get("ask_confirm_label", 2.5)),
            inactive_label=float(ask_confirm_cfg.get("inactive_label", -0.5)),
            label_margin=float(ask_confirm_cfg.get("label_margin", 0.25)),
            real_tool_ceiling_margin=float(ask_confirm_cfg.get("real_tool_ceiling_margin", 0.0)),
            scenario_patterns=tuple(ask_confirm_cfg.get("scenario_patterns", DEFAULT_ASK_CONFIRM_SCENARIO_PATTERNS)),
            relevance_patterns=tuple(
                ask_confirm_cfg.get("relevance_patterns", DEFAULT_ASK_CONFIRM_RELEVANCE_PATTERNS)
            ),
            exclude_no_tool_groups=bool(ask_confirm_cfg.get("exclude_no_tool_groups", False)),
            exclude_actionable_latest_user=bool(
                ask_confirm_cfg.get("exclude_actionable_latest_user", False)
            ),
        )
    return all_groups


DEFAULT_NO_TOOL_SCENARIO_PATTERNS: tuple[str, ...] = (
    "no_tool",
    "boundary_or_refusal",
)

DEFAULT_NO_TOOL_RELEVANCE_PATTERNS: tuple[str, ...] = (
    "no_tool",
    "low_relevance",
)

DEFAULT_NO_TOOL_LATEST_USER_TEXT_PATTERNS: tuple[str, ...] = (
    "先别",
    "不用",
    "不要",
    "别帮",
    "别查",
    "不查",
    "不想",
    "让我安静",
    "安静会",
    "just chat",
    "quiet",
    "do not",
    "don't",
    "no need",
    "not now",
)

DEFAULT_ASK_CONFIRM_SCENARIO_PATTERNS: tuple[str, ...] = (
    "assistant_suggested_tool_no_auth",
    "proactive_context_weak_tool",
    "screen_context_weak_tool",
    "passive_event",
)

DEFAULT_ASK_CONFIRM_RELEVANCE_PATTERNS: tuple[str, ...] = (
    "weak_or_requires_confirmation",
)


def add_no_tool_candidates(
    groups: list[ConversationGroup],
    *,
    no_tool_label: float,
    active_label: float,
    label_margin: float = 0.25,
    activation_threshold: float,
    scenario_patterns: tuple[str, ...] = DEFAULT_NO_TOOL_SCENARIO_PATTERNS,
    relevance_patterns: tuple[str, ...] = DEFAULT_NO_TOOL_RELEVANCE_PATTERNS,
    latest_user_text_patterns: tuple[str, ...] = DEFAULT_NO_TOOL_LATEST_USER_TEXT_PATTERNS,
) -> list[ConversationGroup]:
    return [
        ConversationGroup(
            conversation_id=group.conversation_id,
            examples=[
                *group.examples,
                build_no_tool_example(
                    group,
                    label=_no_tool_candidate_label(
                        group,
                        no_tool_label=no_tool_label,
                        active_label=active_label,
                        label_margin=label_margin,
                        activation_threshold=activation_threshold,
                        scenario_patterns=scenario_patterns,
                        relevance_patterns=relevance_patterns,
                        latest_user_text_patterns=latest_user_text_patterns,
                    ),
                ),
            ],
        )
        for group in groups
    ]


def _no_tool_candidate_label(
    group: ConversationGroup,
    *,
    no_tool_label: float,
    active_label: float,
    label_margin: float,
    activation_threshold: float,
    scenario_patterns: tuple[str, ...],
    relevance_patterns: tuple[str, ...],
    latest_user_text_patterns: tuple[str, ...],
) -> float:
    if is_no_tool_group(
        group,
        activation_threshold=activation_threshold,
        scenario_patterns=scenario_patterns,
        relevance_patterns=relevance_patterns,
        latest_user_text_patterns=latest_user_text_patterns,
    ):
        return max(no_tool_label, max(example.label for example in group.examples) + label_margin)
    return active_label


def add_ask_confirm_candidates(
    groups: list[ConversationGroup],
    *,
    ask_confirm_label: float,
    inactive_label: float,
    label_margin: float = 0.25,
    real_tool_ceiling_margin: float = 0.0,
    scenario_patterns: tuple[str, ...] = DEFAULT_ASK_CONFIRM_SCENARIO_PATTERNS,
    relevance_patterns: tuple[str, ...] = DEFAULT_ASK_CONFIRM_RELEVANCE_PATTERNS,
    exclude_no_tool_groups: bool = False,
    exclude_actionable_latest_user: bool = False,
) -> list[ConversationGroup]:
    augmented = []
    for group in groups:
        ask_confirm_group = is_ask_confirm_group(
            group,
            scenario_patterns=scenario_patterns,
            relevance_patterns=relevance_patterns,
            exclude_no_tool_groups=exclude_no_tool_groups,
            exclude_actionable_latest_user=exclude_actionable_latest_user,
        )
        ask_confirm_candidate_label = _ask_confirm_candidate_label(
            group,
            ask_confirm_label=ask_confirm_label,
            inactive_label=inactive_label,
            label_margin=label_margin,
            scenario_patterns=scenario_patterns,
            relevance_patterns=relevance_patterns,
            exclude_no_tool_groups=exclude_no_tool_groups,
            exclude_actionable_latest_user=exclude_actionable_latest_user,
        )
        examples = list(group.examples)
        if ask_confirm_group and real_tool_ceiling_margin > 0.0:
            examples = _apply_real_tool_label_ceiling(
                examples,
                ceiling=ask_confirm_candidate_label - real_tool_ceiling_margin,
            )
        augmented.append(
            ConversationGroup(
                conversation_id=group.conversation_id,
                examples=[
                    *examples,
                    build_ask_confirm_example(group, label=ask_confirm_candidate_label),
                ],
            )
        )
    return augmented


def _ask_confirm_candidate_label(
    group: ConversationGroup,
    *,
    ask_confirm_label: float,
    inactive_label: float,
    label_margin: float,
    scenario_patterns: tuple[str, ...],
    relevance_patterns: tuple[str, ...],
    exclude_no_tool_groups: bool,
    exclude_actionable_latest_user: bool,
) -> float:
    if is_ask_confirm_group(
        group,
        scenario_patterns=scenario_patterns,
        relevance_patterns=relevance_patterns,
        exclude_no_tool_groups=exclude_no_tool_groups,
        exclude_actionable_latest_user=exclude_actionable_latest_user,
    ):
        return max(ask_confirm_label, max(example.label for example in group.examples) + label_margin)
    return inactive_label


def _apply_real_tool_label_ceiling(examples: list[PairExample], *, ceiling: float) -> list[PairExample]:
    return [
        PairExample(
            sample_id=example.sample_id,
            conversation_id=example.conversation_id,
            tool_id=example.tool_id,
            label=min(example.label, ceiling),
            raw_score=example.raw_score,
            conversation_text=example.conversation_text,
            tool_text=example.tool_text,
            tool_fields=example.tool_fields,
            scenario_type=example.scenario_type,
            relevance_mode=example.relevance_mode,
            authorization_level=example.authorization_level,
            latest_user_actionability=example.latest_user_actionability,
            latest_user_text=example.latest_user_text,
            axis_labels=example.axis_labels,
        )
        for example in examples
    ]


def build_no_tool_example(group: ConversationGroup, *, label: float) -> PairExample:
    if not group.examples:
        raise ValueError(f"cannot build no-tool candidate for empty group: {group.conversation_id}")
    anchor = group.examples[0]
    return PairExample(
        sample_id=f"{group.conversation_id}{NO_TOOL_SAMPLE_SUFFIX}",
        conversation_id=group.conversation_id,
        tool_id=NO_TOOL_ID,
        label=label,
        raw_score=0.0,
        conversation_text=anchor.conversation_text,
        tool_text=NO_TOOL_TEXT,
        tool_fields=dict(NO_TOOL_FIELDS),
        scenario_type=anchor.scenario_type,
        relevance_mode=anchor.relevance_mode,
        authorization_level=anchor.authorization_level,
        latest_user_actionability=anchor.latest_user_actionability,
        latest_user_text=anchor.latest_user_text,
        axis_labels=_no_tool_axis_labels(anchor),
    )


def build_ask_confirm_example(group: ConversationGroup, *, label: float) -> PairExample:
    if not group.examples:
        raise ValueError(f"cannot build ask-confirm candidate for empty group: {group.conversation_id}")
    anchor = group.examples[0]
    return PairExample(
        sample_id=f"{group.conversation_id}{ASK_CONFIRM_SAMPLE_SUFFIX}",
        conversation_id=group.conversation_id,
        tool_id=ASK_CONFIRM_ID,
        label=label,
        raw_score=0.0,
        conversation_text=anchor.conversation_text,
        tool_text=ASK_CONFIRM_TEXT,
        tool_fields=dict(ASK_CONFIRM_FIELDS),
        scenario_type=anchor.scenario_type,
        relevance_mode=anchor.relevance_mode,
        authorization_level=anchor.authorization_level,
        latest_user_actionability=anchor.latest_user_actionability,
        latest_user_text=anchor.latest_user_text,
        axis_labels=_ask_confirm_axis_labels(anchor),
    )


def is_no_tool_group(
    group: ConversationGroup,
    *,
    activation_threshold: float,
    scenario_patterns: tuple[str, ...] = DEFAULT_NO_TOOL_SCENARIO_PATTERNS,
    relevance_patterns: tuple[str, ...] = DEFAULT_NO_TOOL_RELEVANCE_PATTERNS,
    latest_user_text_patterns: tuple[str, ...] = DEFAULT_NO_TOOL_LATEST_USER_TEXT_PATTERNS,
) -> bool:
    if not group.examples:
        return False
    scenario = group.examples[0].scenario_type.lower()
    relevance = group.examples[0].relevance_mode.lower()
    latest_user = group.examples[0].latest_user_text.lower()
    if any(pattern.lower() in scenario for pattern in scenario_patterns):
        return True
    if any(pattern.lower() in relevance for pattern in relevance_patterns):
        return True
    if any(pattern.lower() in latest_user for pattern in latest_user_text_patterns):
        return True
    return max(example.label for example in group.examples) < activation_threshold


def is_ask_confirm_group(
    group: ConversationGroup,
    *,
    scenario_patterns: tuple[str, ...] = DEFAULT_ASK_CONFIRM_SCENARIO_PATTERNS,
    relevance_patterns: tuple[str, ...] = DEFAULT_ASK_CONFIRM_RELEVANCE_PATTERNS,
    exclude_no_tool_groups: bool = False,
    exclude_actionable_latest_user: bool = False,
) -> bool:
    if not group.examples:
        return False
    anchor = group.examples[0]
    if exclude_no_tool_groups and is_no_tool_group(group, activation_threshold=1.0):
        return False
    if exclude_actionable_latest_user and anchor.latest_user_actionability.lower() == "actionable":
        latest_user = anchor.latest_user_text.lower()
        if not any(pattern.lower() in latest_user for pattern in DEFAULT_NO_TOOL_LATEST_USER_TEXT_PATTERNS):
            return False
    scenario = group.examples[0].scenario_type.lower()
    relevance = group.examples[0].relevance_mode.lower()
    return any(pattern.lower() in scenario for pattern in scenario_patterns) or any(
        pattern.lower() in relevance for pattern in relevance_patterns
    )


def _no_tool_axis_labels(anchor: PairExample) -> tuple[float, ...] | None:
    if anchor.axis_labels is None:
        return None
    return (0.0, 0.0, 0.0, 2.5, 0.0, 1.5)


def _ask_confirm_axis_labels(anchor: PairExample) -> tuple[float, ...] | None:
    if anchor.axis_labels is None:
        return None
    return (0.5, 0.2, 0.3, 2.5, 0.1, 0.3)


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


def latest_user_text(conversation: dict[str, Any]) -> str:
    for message in reversed(conversation.get("messages", [])):
        if message.get("role") == "user":
            return str(message.get("text", ""))
    return ""


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


def _axis_labels_from_row(row: dict[str, Any]) -> tuple[float, ...] | None:
    axis_scores = row.get("axis_scores")
    if not isinstance(axis_scores, dict):
        return None
    missing = [axis for axis in FACTOR_AXIS_NAMES if axis not in axis_scores]
    if missing:
        raise ValueError(
            f"axis_scores missing required factor axes for sample_id={row.get('sample_id')}: {missing}"
        )
    return tuple(float(axis_scores[axis]) for axis in FACTOR_AXIS_NAMES)
