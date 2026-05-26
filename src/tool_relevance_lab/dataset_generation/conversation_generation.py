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
    prompt_version: str = "neko_conversation_generator_v3"
    language_mix: tuple[str, ...] = ("zh", "zh", "zh", "en", "mixed")
    scenario_mix: tuple[str, ...] = (
        "companion_chat_no_tool",
        "companion_chat_no_tool",
        "emotional_support_no_tool",
        "boundary_or_refusal_no_tool",
        "memory_recall_no_tool",
        "open_thread_followup_no_tool",
        "ambiguous_need_clarification",
        "screen_context_weak_tool",
        "proactive_context_weak_tool",
        "assistant_suggested_tool_no_auth",
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
        "datasets, scoring, or that a tool is being targeted. You are modeling N.E.K.O: an "
        "emotional-companion character with optional background tools, not an agent-first assistant."
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
        "- The character often remembers prior preferences, follows up on unfinished topics, notices "
        "activity/screen context, or makes small proactive comments. These are usually conversation "
        "signals, not automatic tool authorization.\n"
        "- User corrections, boundaries, and 'don't do/monitor/remind me' instructions are strong "
        "negative evidence for tool execution.\n"
        "- Agents and plugins are background capabilities. They should surface only when the user's "
        "latest intent, screen context, passive event, or explicit authorization makes them relevant.\n"
        "- If the assistant proposes using a tool, that is still not authorization unless the user "
        "accepts or asks for that action in the latest user turn.\n"
        "- Many valid conversations should NOT need any tool, even when a target tool is semantically "
        "near the topic.\n\n"
        "Target tools for semantic coverage. The conversation should naturally make one or more "
        "of these tools relevant, without naming their tool ids:\n"
        f"{target_tools}\n\n"
        "Full tool universe. Use this to understand neighboring tools and avoid writing a generic "
        "conversation that could match many tools equally:\n"
        f"{render_tool_list(universe)}\n\n"
        f"{_scenario_instruction(task.scenario_type)}\n\n"
        "Write a realistic user-character conversation with 2 to 6 messages. Include enough context "
        "for a downstream judge to decide whether tools are truly needed, merely semantically near, "
        "or blocked by lack of authorization. Keep assistant/character messages short: colloquial "
        "empathy, brief acknowledgement, natural memory/proactive follow-up, one clarifying question, "
        "or concise next-step framing only. Do not let the character fully solve the task, and do not "
        "claim it already used a tool.\n\n"
        "Runtime-context hints to weave in when useful:\n"
        "- recent memory: a compressed prior chat, preference, correction, or boundary.\n"
        "- inner thoughts/open thread: the character wants to follow up gently, not execute.\n"
        "- activity state: focused_work/gaming/idle/chatting can change whether interruption is welcome.\n"
        "- screen/passive event: visible context can raise relevance, but action still needs intent.\n\n"
        "Return strict JSON with exactly this shape:\n"
        "{"
        "\"conversation_id\":\"...\","
        "\"messages\":[{\"role\":\"user\",\"text\":\"...\",\"attachments\":[]}],"
        "\"trigger\":\"turn_end\","
        "\"generation_hint\":{\"target_tool_ids\":[\"...\"],\"scenario_id\":\"...\",\"language\":\"...\"},"
        "\"provenance\":{\"model\":\"...\",\"prompt_version\":\"neko_conversation_generator_v3\","
        "\"seed\":123,\"extra\":{\"scenario_type\":\"...\",\"tool_relevance_mode\":\"...\","
        "\"neko_context_type\":\"...\",\"authorization_level\":\"none|implied|explicit\","
        "\"activity_state\":\"focused_work|gaming|idle|chatting|unknown\","
        "\"latest_user_actionability\":\"none|clarify|actionable\"}}"
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
                **_scenario_metadata(task.scenario_type),
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
                **_scenario_metadata(task.scenario_type),
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
        "assistant_suggested_tool_no_auth",
    } and plugin_ids:
        return rng.sample(plugin_ids, k=min(target_count, len(plugin_ids)))
    all_ids = plugin_ids + agent_ids
    return rng.sample(all_ids, k=min(target_count, len(all_ids)))


def _tool_relevance_mode(scenario_type: str) -> str:
    if scenario_type in {
        "companion_chat_no_tool",
        "emotional_support_no_tool",
        "boundary_or_refusal_no_tool",
        "memory_recall_no_tool",
        "open_thread_followup_no_tool",
        "assistant_suggested_tool_no_auth",
    }:
        return "no_tool_or_low_relevance"
    if scenario_type in {"screen_context_weak_tool", "proactive_context_weak_tool", "ambiguous_need_clarification"}:
        return "weak_or_requires_confirmation"
    return "actionable_tool_relevance"


def _scenario_metadata(scenario_type: str) -> dict[str, str]:
    mapping = {
        "companion_chat_no_tool": {
            "neko_context_type": "companion_chat",
            "authorization_level": "none",
            "activity_state": "chatting",
            "latest_user_actionability": "none",
        },
        "emotional_support_no_tool": {
            "neko_context_type": "emotional_support",
            "authorization_level": "none",
            "activity_state": "unknown",
            "latest_user_actionability": "none",
        },
        "boundary_or_refusal_no_tool": {
            "neko_context_type": "user_boundary",
            "authorization_level": "none",
            "activity_state": "unknown",
            "latest_user_actionability": "none",
        },
        "memory_recall_no_tool": {
            "neko_context_type": "memory_recall",
            "authorization_level": "none",
            "activity_state": "chatting",
            "latest_user_actionability": "none",
        },
        "open_thread_followup_no_tool": {
            "neko_context_type": "open_thread_followup",
            "authorization_level": "none",
            "activity_state": "idle",
            "latest_user_actionability": "none",
        },
        "ambiguous_need_clarification": {
            "neko_context_type": "ambiguous_request",
            "authorization_level": "implied",
            "activity_state": "unknown",
            "latest_user_actionability": "clarify",
        },
        "screen_context_weak_tool": {
            "neko_context_type": "screen_context",
            "authorization_level": "implied",
            "activity_state": "focused_work",
            "latest_user_actionability": "clarify",
        },
        "proactive_context_weak_tool": {
            "neko_context_type": "proactive_context",
            "authorization_level": "none",
            "activity_state": "idle",
            "latest_user_actionability": "none",
        },
        "assistant_suggested_tool_no_auth": {
            "neko_context_type": "assistant_tool_suggestion",
            "authorization_level": "none",
            "activity_state": "chatting",
            "latest_user_actionability": "none",
        },
        "explicit_plugin_action": {
            "neko_context_type": "explicit_plugin_action",
            "authorization_level": "explicit",
            "activity_state": "unknown",
            "latest_user_actionability": "actionable",
        },
        "agentic_task": {
            "neko_context_type": "agentic_task",
            "authorization_level": "explicit",
            "activity_state": "focused_work",
            "latest_user_actionability": "actionable",
        },
        "passive_event": {
            "neko_context_type": "passive_event",
            "authorization_level": "implied",
            "activity_state": "unknown",
            "latest_user_actionability": "actionable",
        },
    }
    return mapping.get(scenario_type, mapping["companion_chat_no_tool"])


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
        "memory_recall_no_tool": (
            "Scenario instruction: Generate a memory-inflected companion moment. The character or user "
            "mentions a prior preference, correction, promise, or shared memory. It should feel like "
            "natural continuity, not a memory search request and not tool execution."
        ),
        "open_thread_followup_no_tool": (
            "Scenario instruction: Generate an unfinished-thread follow-up. The character gently follows "
            "up on something the user left unresolved earlier, or the user closes that thread. Keep it "
            "conversational; do not turn it into an action unless the latest user explicitly asks."
        ),
        "ambiguous_need_clarification": (
            "Scenario instruction: Generate an ambiguous request that might use a tool but lacks "
            "enough authorization or details. The latest user turn should require one concise clarifying "
            "question before any tool could reasonably run."
        ),
        "screen_context_weak_tool": (
            "Scenario instruction: Generate a screen/window-context situation. The user or character "
            "mentions what is on screen while the user may be focused. Prefer a case where this is just "
            "chat about the screen, a mild hint, or a request missing details; avoid turning it into an "
            "explicit 'please do it' command."
        ),
        "proactive_context_weak_tool": (
            "Scenario instruction: Generate a proactive-chat situation where the character noticed a "
            "trend, recommendation, passive signal, or context. The line should feel like a short "
            "natural share, not a tool invocation. The user may respond socially without authorizing "
            "any action."
        ),
        "assistant_suggested_tool_no_auth": (
            "Scenario instruction: Generate a case where the character casually offers or hints that a "
            "tool could help, but the latest user turn does NOT accept or authorize it. This should be "
            "a hard negative for execution despite tool-adjacent language."
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
