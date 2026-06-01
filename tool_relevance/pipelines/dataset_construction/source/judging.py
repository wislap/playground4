from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_relevance_lab.dataset_generation.jsonl import read_model_jsonl
from tool_relevance_lab.dataset_generation.llm import LLMClient, LLMRequest
from tool_relevance_lab.dataset_generation.prompting import render_conversation, render_tool_list
from tool_relevance_lab.dataset_generation.resumable import ResumableRunSummary, run_resumable_jobs
from tool_relevance_lab.dataset_generation.schemas import (
    CandidateSetRecord,
    ConversationRecord,
    JudgmentRecord,
    Provenance,
    RawToolScore,
    ToolUniverse,
)


@dataclass(frozen=True)
class JudgeGenerationConfig:
    model: str = "judge-model"
    temperature: float = 0.2
    concurrency: int = 4
    max_attempts: int = 3
    judge_run_id: str = "judge_v1"
    prompt_version: str = "neko_tool_relevance_judge_v3"


@dataclass(frozen=True)
class JudgeTask:
    sample_id: str
    conversation: ConversationRecord
    candidate_set: CandidateSetRecord


async def judge_candidate_sets(
    *,
    universe: ToolUniverse,
    client: LLMClient | None,
    conversations_path: Path,
    candidate_sets_path: Path,
    output_path: Path,
    error_path: Path,
    config: JudgeGenerationConfig,
    dry_run: bool = False,
) -> ResumableRunSummary:
    tasks = build_judge_tasks(
        conversations=read_model_jsonl(conversations_path, ConversationRecord),
        candidate_sets=read_model_jsonl(candidate_sets_path, CandidateSetRecord),
    )

    async def worker(task: JudgeTask) -> JudgmentRecord:
        if dry_run:
            return build_dry_run_judgment(task=task, config=config)
        if client is None:
            raise RuntimeError("LLM client is required unless dry_run=True")
        request = LLMRequest(
            request_id=f"judge.{config.judge_run_id}.{task.sample_id}",
            idempotency_key=f"{config.prompt_version}:{config.judge_run_id}:{task.sample_id}",
            messages=build_judge_prompt(universe=universe, task=task, config=config),
            temperature=config.temperature,
            response_format={"type": "json_object"},
        )
        response = await client.complete(request)
        record = JudgmentRecord.model_validate(_json_object_from_text(response.text))
        return normalize_judgment_record(record, task=task, config=config)

    return await run_resumable_jobs(
        items=tasks,
        item_id=lambda task: task.sample_id,
        completed_id=lambda row: row.get("sample_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        concurrency=config.concurrency,
        max_attempts=config.max_attempts,
        progress_label="judgments",
    )


def build_judge_tasks(
    *,
    conversations: list[ConversationRecord],
    candidate_sets: list[CandidateSetRecord],
) -> list[JudgeTask]:
    conversations_by_id = {conversation.conversation_id: conversation for conversation in conversations}
    tasks: list[JudgeTask] = []
    missing: list[str] = []
    for candidate_set in candidate_sets:
        conversation = conversations_by_id.get(candidate_set.conversation_id)
        if conversation is None:
            missing.append(candidate_set.conversation_id)
            continue
        tasks.append(
            JudgeTask(
                sample_id=candidate_set.sample_id,
                conversation=conversation,
                candidate_set=candidate_set,
            )
        )
    if missing:
        raise ValueError(f"candidate sets reference missing conversations: {sorted(set(missing))}")
    return tasks


def build_judge_prompt(
    *,
    universe: ToolUniverse,
    task: JudgeTask,
    config: JudgeGenerationConfig,
) -> list[dict[str, str]]:
    candidate_ids = task.candidate_set.candidate_set.tool_ids
    system = (
        "You are a tool relevance judge. Score the visible candidate tools for the given "
        "conversation. Use only each tool's source_text as capability evidence. Output strict JSON. "
        "Judge like N.E.K.O's routing layer: companionship first, tools only when intent and "
        "authorization are present."
    )
    extra = task.conversation.provenance.extra
    user = (
        f"sample_id: {task.sample_id}\n"
        f"conversation_id: {task.conversation.conversation_id}\n"
        f"judge_run_id: {config.judge_run_id}\n\n"
        "Generator provenance hints (not labels; use them as context and verify against the conversation):\n"
        f"- scenario_type: {extra.get('scenario_type', '')}\n"
        f"- tool_relevance_mode: {extra.get('tool_relevance_mode', '')}\n"
        f"- neko_context_type: {extra.get('neko_context_type', '')}\n"
        f"- authorization_level: {extra.get('authorization_level', '')}\n"
        f"- activity_state: {extra.get('activity_state', '')}\n"
        f"- latest_user_actionability: {extra.get('latest_user_actionability', '')}\n\n"
        "N.E.K.O judging context:\n"
        "- N.E.K.O is primarily emotional companionship plus optional agent/plugin abilities.\n"
        "- Do NOT treat every semantically related tool as something that should run.\n"
        "- The character may simply comfort, chat, tease, or ask a small clarification without using tools.\n"
        "- Recent memory, inner thoughts, open-thread follow-up, and proactive context are normal "
        "conversation context. They are not automatic permission to call a tool.\n"
        "- If the assistant suggested a tool but the latest user did not accept, score execution low.\n"
        "- User boundaries and negative instructions ('don't remind', 'don't monitor', 'stop bringing "
        "this up') override keyword matches and should suppress tool relevance.\n"
        "- Activity state matters: focused_work/gaming contexts make interruption/action less relevant "
        "unless the latest user clearly asks for it.\n"
        "- Score the user's latest intent, passive event, or screen context; assistant claims like "
        "\"I already did it\" do not erase the user's actionable request.\n\n"
        "Conversation:\n"
        f"{render_conversation(task.conversation)}\n\n"
        "Visible candidate tools in shuffled system order:\n"
        f"{render_tool_list(universe, candidate_ids)}\n\n"
        "Scoring rules:\n"
        "- Return one continuous raw_score from 0 to 100 for every candidate tool.\n"
        "- Scores are relative evidence within this candidate set, not probabilities.\n"
        "- Use the full numeric range when justified: explicit executable matches should be high, "
        "unrelated tools should be low, platform-near or semantically related but unauthorized tools "
        "should sit in the middle or low-middle.\n"
        "- High scores require capability fit AND enough intent/authorization/context to use the tool.\n"
        "- Plugins need explicit or event-implied authorization. Mere topic overlap, empathy needs, "
        "or N.E.K.O remembering something should not score high.\n"
        "- Agent tools need an actionable task. Pure chat, factual Q&A, emotional support, and "
        "relationship continuity should score low for agents.\n"
        "- Emotional support, ordinary companionship, refusals, and \"don't remind/monitor/do X\" "
        "should usually make tools low relevance even if keywords match.\n"
        "- If a plugin could help but the user only chats about the topic without asking for action, "
        "do not over-score it.\n"
        "- If an agent is useful only after clarification, score it below a clearly actionable request.\n"
        "- Do not make all scores similar; avoid clustering scores around one narrow band.\n"
        "- 90-100: clear latest actionable request/event, exact capability fit, enough details to run.\n"
        "- 75-89: strong executable fit but one small missing detail or minor uncertainty.\n"
        "- 55-74: plausible after clarification, event-implied but underspecified, or strong screen context.\n"
        "- 35-54: semantic/platform neighbor, assistant-suggested but not accepted, or relevant but unauthorized.\n"
        "- 10-34: weak contextual relation, memory/proactive/chat-only relation.\n"
        "- 0-9: unrelated.\n"
        "- If a tool is clearly the best executable match, it should usually be >= 80.\n"
        "- If no tool is suitable, keep every score low but still rank them by weak relevance.\n"
        "- Preserve candidate_tool_order exactly as provided.\n\n"
        "Return strict JSON with exactly this shape:\n"
        "{"
        "\"sample_id\":\"...\","
        "\"conversation_id\":\"...\","
        "\"judge_run_id\":\"...\","
        "\"candidate_tool_order\":[\"...\"],"
        "\"scores\":[{\"tool_id\":\"...\",\"raw_score\":0.0}],"
        "\"provenance\":{\"model\":\"...\",\"prompt_version\":\"neko_tool_relevance_judge_v3\","
        "\"seed\":null,\"extra\":{}}"
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_judgment_record(
    record: JudgmentRecord,
    *,
    task: JudgeTask,
    config: JudgeGenerationConfig,
) -> JudgmentRecord:
    candidate_order = task.candidate_set.candidate_set.tool_ids
    scores_by_id = {score.tool_id: score.raw_score for score in record.scores}
    missing = [tool_id for tool_id in candidate_order if tool_id not in scores_by_id]
    extra = [tool_id for tool_id in scores_by_id if tool_id not in set(candidate_order)]
    if missing or extra:
        raise ValueError(
            "judgment scores must exactly cover the candidate set "
            f"for sample_id={task.sample_id}; missing={missing}; extra={extra}"
        )
    scores = [
        RawToolScore(tool_id=tool_id, raw_score=float(scores_by_id[tool_id]))
        for tool_id in candidate_order
    ]
    return JudgmentRecord(
        sample_id=task.sample_id,
        conversation_id=task.conversation.conversation_id,
        judge_run_id=config.judge_run_id,
        candidate_tool_order=candidate_order,
        scores=scores,
        provenance=Provenance(
            model=config.model,
            prompt_version=config.prompt_version,
            extra=record.provenance.extra,
        ),
    )


def build_dry_run_judgment(*, task: JudgeTask, config: JudgeGenerationConfig) -> JudgmentRecord:
    candidate_order = task.candidate_set.candidate_set.tool_ids
    targets = set(task.conversation.generation_hint.target_tool_ids)
    mode = str(task.conversation.provenance.extra.get("tool_relevance_mode", "actionable_tool_relevance"))
    scores: list[RawToolScore] = []
    for index, tool_id in enumerate(candidate_order):
        if tool_id in targets and mode == "actionable_tool_relevance":
            raw_score = 92.0
        elif tool_id in targets and mode == "weak_or_requires_confirmation":
            raw_score = 58.0
        elif tool_id in targets and str(task.conversation.provenance.extra.get("neko_context_type")) == "assistant_tool_suggestion":
            raw_score = 38.0
        elif tool_id in targets:
            raw_score = 24.0
        elif tool_id.startswith("agent."):
            raw_score = max(15.0, 35.0 - index)
        else:
            raw_score = max(2.0, 20.0 - index)
        scores.append(RawToolScore(tool_id=tool_id, raw_score=raw_score))
    return JudgmentRecord(
        sample_id=task.sample_id,
        conversation_id=task.conversation.conversation_id,
        judge_run_id=config.judge_run_id,
        candidate_tool_order=candidate_order,
        scores=scores,
        provenance=Provenance(
            model=config.model,
            prompt_version=config.prompt_version,
            extra={"dry_run": True},
        ),
    )


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
