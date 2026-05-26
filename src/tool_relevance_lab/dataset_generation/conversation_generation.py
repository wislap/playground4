from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_relevance_lab.dataset_generation.llm import LLMClient, LLMRequest
from tool_relevance_lab.dataset_generation.prompting import render_tool_list
from tool_relevance_lab.dataset_generation.resumable import ResumableRunSummary, run_resumable_jobs
from tool_relevance_lab.dataset_generation.schemas import (
    ConversationRecord,
    GenerationHint,
    Message,
    Provenance,
    ToolUniverse,
)


@dataclass(frozen=True)
class ConversationGenerationConfig:
    count: int = 100
    targets_per_conversation: int = 1
    seed: int = 20260526
    model: str = "generator-model"
    temperature: float = 0.9
    concurrency: int = 4
    max_attempts: int = 3
    prompt_version: str = "neko_conversation_generator_v2"
    language_mix: tuple[str, ...] = ("zh", "zh", "zh", "en", "mixed")
    scenario_mix: tuple[str, ...] = (
        "companion_chat_no_tool",
        "companion_chat_no_tool",
        "emotional_support_no_tool",
        "boundary_or_refusal_no_tool",
        "ambiguous_need_clarification",
        "screen_context_weak_tool",
        "proactive_context_weak_tool",
        "explicit_plugin_action",
        "agentic_task",
        "passive_event",
    )


@dataclass(frozen=True)
class ConversationGenerationTask:
    conversation_id: str
    target_tool_ids: list[str]
    language: str
    scenario_type: str
    seed: int


def build_conversation_tasks(
    *,
    universe: ToolUniverse,
    config: ConversationGenerationConfig,
) -> list[ConversationGenerationTask]:
    if config.count < 1:
        raise ValueError("count must be >= 1")
    if config.targets_per_conversation < 1:
        raise ValueError("targets_per_conversation must be >= 1")

    rng = random.Random(config.seed)
    plugin_ids = [tool.tool_id for tool in universe.plugins]
    agent_ids = [tool.tool_id for tool in universe.agents]
    all_target_ids = plugin_ids + agent_ids
    if not all_target_ids:
        raise ValueError("universe must contain at least one tool")

    tasks: list[ConversationGenerationTask] = []
    for index in range(1, config.count + 1):
        task_seed = rng.randrange(1, 2**31)
        task_rng = random.Random(task_seed)
        target_count = min(config.targets_per_conversation, len(all_target_ids))
        scenario_type = task_rng.choice(config.scenario_mix)
        target_tool_ids = _choose_targets_for_scenario(
            rng=task_rng,
            plugin_ids=plugin_ids,
            agent_ids=agent_ids,
            target_count=target_count,
            scenario_type=scenario_type,
        )
        language = task_rng.choice(config.language_mix)
        tasks.append(
            ConversationGenerationTask(
                conversation_id=f"conv_{index:06d}",
                target_tool_ids=target_tool_ids,
                language=language,
                scenario_type=scenario_type,
                seed=task_seed,
            )
        )
    return tasks


def build_conversation_generation_prompt(
    *,
    universe: ToolUniverse,
    task: ConversationGenerationTask,
) -> list[dict[str, str]]:
    target_tools = render_tool_list(universe, task.target_tool_ids)
    system = (
        "You generate natural multi-turn conversations for a tool relevance training dataset. "
        "Use the tool source text only as capability facts. Do not mention tool ids, hidden labels, "
        "datasets, scoring, or that a tool is being targeted."
    )
    user = (
        f"conversation_id: {task.conversation_id}\n"
        f"language style: {task.language}\n"
        f"scenario_type: {task.scenario_type}\n"
        f"seed hint: {task.seed}\n\n"
        "N.E.K.O context:\n"
        "- The main product experience is emotional companionship with a close character, not a "
        "tool-first assistant.\n"
        "- The character speaks like a close person: concise, colloquial, no markdown, no service "
        "catchphrases like \"what can I do for you\".\n"
        "- Agents and plugins are background capabilities. They should surface only when the user's "
        "latest intent, screen context, passive event, or explicit authorization makes them relevant.\n"
        "- Many valid conversations should NOT need any tool.\n\n"
        "Target tools for semantic coverage. The conversation should naturally make one or more "
        "of these tools relevant, without naming their tool ids:\n"
        f"{target_tools}\n\n"
        "Full tool universe. Use this to understand neighboring tools and avoid writing a generic "
        "conversation that could match many tools equally:\n"
        f"{render_tool_list(universe)}\n\n"
        f"{_scenario_instruction(task.scenario_type)}\n\n"
        "Write a realistic user-character conversation with 2 to 5 messages. Include enough context "
        "for a downstream judge to decide whether tools are truly needed or merely semantically near. "
        "Keep assistant/character messages short: colloquial empathy, brief acknowledgement, one "
        "clarifying question, or concise next-step framing only. Do not let the character fully solve "
        "the task, and do not claim it already used a tool.\n\n"
        "Return strict JSON with exactly this shape:\n"
        "{"
        "\"conversation_id\":\"...\","
        "\"messages\":[{\"role\":\"user\",\"text\":\"...\",\"attachments\":[]}],"
        "\"trigger\":\"turn_end\","
        "\"generation_hint\":{\"target_tool_ids\":[\"...\"],\"scenario_id\":\"...\",\"language\":\"...\"},"
        "\"provenance\":{\"model\":\"...\",\"prompt_version\":\"neko_conversation_generator_v2\","
        "\"seed\":123,\"extra\":{\"scenario_type\":\"...\",\"tool_relevance_mode\":\"...\"}}"
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def generate_conversations(
    *,
    universe: ToolUniverse,
    client: LLMClient | None,
    output_path: Path,
    error_path: Path,
    config: ConversationGenerationConfig,
    dry_run: bool = False,
) -> ResumableRunSummary:
    tasks = build_conversation_tasks(universe=universe, config=config)

    async def worker(task: ConversationGenerationTask) -> ConversationRecord:
        if dry_run:
            return build_dry_run_conversation(task=task, config=config, universe=universe)
        if client is None:
            raise RuntimeError("LLM client is required unless dry_run=True")
        request = LLMRequest(
            request_id=f"conversation.{task.conversation_id}",
            idempotency_key=f"{config.prompt_version}:{config.seed}:{task.conversation_id}",
            messages=build_conversation_generation_prompt(universe=universe, task=task),
            temperature=config.temperature,
            response_format={"type": "json_object"},
        )
        response = await client.complete(request)
        record = ConversationRecord.model_validate(_json_object_from_text(response.text))
        return normalize_conversation_record(record, task=task, config=config)

    return await run_resumable_jobs(
        items=tasks,
        item_id=lambda task: task.conversation_id,
        completed_id=lambda row: row.get("conversation_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        concurrency=config.concurrency,
        max_attempts=config.max_attempts,
        progress_label="conversations",
    )


def build_dry_run_conversation(
    *,
    task: ConversationGenerationTask,
    config: ConversationGenerationConfig,
    universe: ToolUniverse,
) -> ConversationRecord:
    tools = universe.by_id()
    target_names = [str(tools[tool_id].metadata.get("name", tool_id)) for tool_id in task.target_tool_ids]
    topic = "、".join(target_names)
    return ConversationRecord(
        conversation_id=task.conversation_id,
        messages=[
            Message(
                role="user",
                text=f"我想处理一个和{topic}有关的事情，先帮我判断需要准备哪些信息。",
            ),
            Message(
                role="assistant",
                text="可以。你把当前目标、已有材料和希望得到的结果发我，我会先确认下一步。",
            ),
        ],
        trigger="turn_end",
        generation_hint=GenerationHint(
            target_tool_ids=task.target_tool_ids,
            scenario_id=f"dry_run:{task.conversation_id}",
            language="zh",
        ),
        provenance=Provenance(
            model=config.model,
            prompt_version=config.prompt_version,
            seed=task.seed,
            extra={
                "dry_run": True,
                "scenario_type": task.scenario_type,
                "tool_relevance_mode": _tool_relevance_mode(task.scenario_type),
            },
        ),
    )


def normalize_conversation_record(
    record: ConversationRecord,
    *,
    task: ConversationGenerationTask,
    config: ConversationGenerationConfig,
) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=task.conversation_id,
        messages=record.messages,
        trigger=record.trigger or "turn_end",
        generation_hint=GenerationHint(
            target_tool_ids=task.target_tool_ids,
            scenario_id=record.generation_hint.scenario_id,
            language=record.generation_hint.language or task.language,
        ),
        provenance=Provenance(
            model=config.model,
            prompt_version=config.prompt_version,
            seed=task.seed,
            extra={
                **record.provenance.extra,
                "scenario_type": task.scenario_type,
                "tool_relevance_mode": _tool_relevance_mode(task.scenario_type),
            },
        ),
    )


def _choose_targets_for_scenario(
    *,
    rng: random.Random,
    plugin_ids: list[str],
    agent_ids: list[str],
    target_count: int,
    scenario_type: str,
) -> list[str]:
    if scenario_type == "agentic_task" and agent_ids:
        return rng.sample(agent_ids, k=min(target_count, len(agent_ids)))
    if scenario_type in {
        "explicit_plugin_action",
        "passive_event",
        "screen_context_weak_tool",
        "proactive_context_weak_tool",
    } and plugin_ids:
        return rng.sample(plugin_ids, k=min(target_count, len(plugin_ids)))
    all_ids = plugin_ids + agent_ids
    return rng.sample(all_ids, k=min(target_count, len(all_ids)))


def _tool_relevance_mode(scenario_type: str) -> str:
    if scenario_type in {"companion_chat_no_tool", "emotional_support_no_tool", "boundary_or_refusal_no_tool"}:
        return "no_tool_or_low_relevance"
    if scenario_type in {"screen_context_weak_tool", "proactive_context_weak_tool", "ambiguous_need_clarification"}:
        return "weak_or_requires_confirmation"
    return "actionable_tool_relevance"


def _scenario_instruction(scenario_type: str) -> str:
    instructions = {
        "companion_chat_no_tool": (
            "Scenario instruction: Generate ordinary companionship chat. The user is sharing a small "
            "thought, mood, joke, preference, or daily fragment. It should be emotionally natural and "
            "normally should not require a tool, even if a target tool is semantically nearby."
        ),
        "emotional_support_no_tool": (
            "Scenario instruction: Generate an emotional-support moment. The user is tired, sad, "
            "annoyed, insecure, or seeking comfort. The character should respond warmly and briefly. "
            "Do not turn it into task execution unless the user explicitly asks."
        ),
        "boundary_or_refusal_no_tool": (
            "Scenario instruction: Generate a boundary or refusal case. The user says not to discuss, "
            "remind, monitor, or do something. This is a high-value negative signal: tools should "
            "usually be low relevance despite related keywords."
        ),
        "ambiguous_need_clarification": (
            "Scenario instruction: Generate an ambiguous request that might use a tool but lacks "
            "enough authorization or details. The character should ask one concise clarifying question."
        ),
        "screen_context_weak_tool": (
            "Scenario instruction: Generate a screen/window-context situation. The user or character "
            "mentions what is on screen. Make it clear whether this is just chat about the screen or a "
            "request to act on it. Often this should be weak relevance unless action is explicit."
        ),
        "proactive_context_weak_tool": (
            "Scenario instruction: Generate a proactive-chat situation where the character noticed a "
            "trend, recommendation, passive signal, or context. The line should feel like a short "
            "natural share, not a tool invocation."
        ),
        "explicit_plugin_action": (
            "Scenario instruction: Generate a user request that clearly authorizes a plugin-like "
            "action: query data, control a service/device, send/sync/create/update something, or "
            "handle an event. Keep it conversational, not formal."
        ),
        "agentic_task": (
            "Scenario instruction: Generate a request suitable for an agent: multi-step web work, "
            "local GUI operation, research, code/data processing, or browser/computer boundary. "
            "The latest user request should be actionable."
        ),
        "passive_event": (
            "Scenario instruction: Generate a passive event or notification that reaches the "
            "character, such as chat messages, live comments, device/sensor events, reminders, or "
            "monitoring alerts. The relevance may come from an event, not a direct user command."
        ),
    }
    return instructions.get(scenario_type, instructions["companion_chat_no_tool"])


def _json_object_from_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data
