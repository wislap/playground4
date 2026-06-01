from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tool_relevance_lab.dataset_generation.jsonl import read_model_jsonl
from tool_relevance_lab.dataset_generation.resumable import ResumableRunSummary, run_resumable_jobs
from tool_relevance_lab.dataset_generation.sampling import CandidateSampler, CandidateSamplingConfig
from tool_relevance_lab.dataset_generation.schemas import CandidateSetRecord, ConversationRecord, ToolUniverse


@dataclass(frozen=True)
class CandidateGenerationConfig:
    plugin_sample_rate: float = 0.10
    target_force_rate: float = 1.0
    weight_decay_on_select: float = 0.70
    seed: int = 20260526
    concurrency: int = 1


@dataclass(frozen=True)
class CandidateGenerationTask:
    sample_id: str
    conversation: ConversationRecord
    seed: int
    record: CandidateSetRecord | None = None


def build_candidate_tasks(
    conversations: list[ConversationRecord],
    *,
    seed: int,
) -> list[CandidateGenerationTask]:
    return [
        CandidateGenerationTask(
            sample_id=f"sample_{index:06d}",
            conversation=conversation,
            seed=seed + index,
        )
        for index, conversation in enumerate(conversations, start=1)
    ]


async def generate_candidate_sets(
    *,
    universe: ToolUniverse,
    conversations_path: Path,
    output_path: Path,
    error_path: Path,
    config: CandidateGenerationConfig,
) -> ResumableRunSummary:
    conversations = read_model_jsonl(conversations_path, ConversationRecord)
    tasks = _materialize_candidate_tasks(
        universe=universe,
        conversations=conversations,
        config=config,
    )

    async def worker(task: CandidateGenerationTask) -> CandidateSetRecord:
        if task.record is None:
            raise RuntimeError(f"candidate task was not materialized: {task.sample_id}")
        return task.record

    return await run_resumable_jobs(
        items=tasks,
        item_id=lambda task: task.sample_id,
        completed_id=lambda row: row.get("sample_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        concurrency=config.concurrency,
        max_attempts=1,
        progress_label="candidate_sets",
    )


def _materialize_candidate_tasks(
    *,
    universe: ToolUniverse,
    conversations: list[ConversationRecord],
    config: CandidateGenerationConfig,
) -> list[CandidateGenerationTask]:
    sampler = CandidateSampler(
        universe,
        CandidateSamplingConfig(
            plugin_sample_rate=config.plugin_sample_rate,
            target_force_rate=config.target_force_rate,
            weight_decay_on_select=config.weight_decay_on_select,
            shuffle_candidates=True,
        ),
    )
    tasks = build_candidate_tasks(conversations, seed=config.seed)
    materialized: list[CandidateGenerationTask] = []
    for task in tasks:
        record = sampler.sample(
            sample_id=task.sample_id,
            conversation_id=task.conversation.conversation_id,
            seed=task.seed,
            target_tool_ids=task.conversation.generation_hint.target_tool_ids,
        )
        materialized.append(
            CandidateGenerationTask(
                sample_id=task.sample_id,
                conversation=task.conversation,
                seed=task.seed,
                record=record,
            )
        )
    return materialized
