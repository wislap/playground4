from __future__ import annotations

from tool_relevance_lab.dataset_generation.schemas import (
    CandidateSetRecord,
    ConversationRecord,
    ToolClassification,
    ToolUniverse,
)


def render_tool_list(universe: ToolUniverse, tool_ids: list[str] | None = None) -> str:
    tools_by_id = universe.by_id()
    selected = tool_ids or [tool.tool_id for tool in universe.tools]
    parts: list[str] = []
    for index, tool_id in enumerate(selected, start=1):
        tool = tools_by_id[tool_id]
        parts.append(f"[{index}] tool_id: {tool.tool_id}\nkind: {tool.kind}\n{tool.source_text}")
    return "\n\n".join(parts)


def render_conversation(conversation: ConversationRecord) -> str:
    lines = [f"trigger: {conversation.trigger}"]
    for message in conversation.messages:
        attachment_note = ""
        if message.attachments:
            attachment_note = f" attachments:{len(message.attachments)}"
        lines.append(f"{message.role}{attachment_note}: {message.text}")
    return "\n".join(lines)


def build_generator_prompt(universe: ToolUniverse, target_tool_ids: list[str]) -> list[dict[str, str]]:
    system = (
        "Generate natural conversations for a tool relevance dataset. "
        "Use only the provided tool ids and source text as capability facts. "
        "Do not mention target tool ids in the conversation."
    )
    user = (
        "Target tool ids for coverage accounting:\n"
        f"{', '.join(target_tool_ids)}\n\n"
        "Full tool universe:\n"
        f"{render_tool_list(universe)}\n\n"
        "Return one strict JSON object with conversation_id, messages, trigger, "
        "generation_hint, and provenance."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_classification_prompt(universe: ToolUniverse) -> list[dict[str, str]]:
    system = (
        "Classify tools for synthetic dataset generation coverage. "
        "Use only tool_id, kind, and source_text. Do not add capabilities. "
        "The classification is not model input and not production metadata."
    )
    user = (
        "Classify every tool into a primary category, optional secondary categories, "
        "and generation tags. Return strict JSON matching tool_universe_id, "
        "taxonomy_version, and categories.\n\n"
        f"Tools:\n{render_tool_list(universe)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_classification_summary(classification: ToolClassification) -> str:
    lines = [
        f"tool_universe_id: {classification.tool_universe_id}",
        f"taxonomy_version: {classification.taxonomy_version}",
    ]
    for category in classification.categories:
        tags = ", ".join(category.generation_tags)
        secondary = ", ".join(category.secondary_categories)
        lines.append(
            f"- {category.tool_id}: primary={category.primary_category}; "
            f"secondary={secondary}; tags={tags}"
        )
    return "\n".join(lines)


def build_judge_prompt(
    universe: ToolUniverse,
    conversation: ConversationRecord,
    candidate_record: CandidateSetRecord,
) -> list[dict[str, str]]:
    system = (
        "Judge relative tool relevance for the visible candidate set. "
        "Score every candidate from 0 to 100. Raw scores are relative evidence, "
        "not final probabilities. Do not infer capabilities absent from source_text."
    )
    user = (
        "Conversation:\n"
        f"{render_conversation(conversation)}\n\n"
        "Candidate tools in shuffled system order:\n"
        f"{render_tool_list(universe, candidate_record.candidate_set.tool_ids)}\n\n"
        "Return strict JSON with sample_id, conversation_id, judge_run_id, "
        "candidate_tool_order, scores, and provenance. If at least one candidate "
        "is suitable, the best suitable tool should usually be above 80."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
