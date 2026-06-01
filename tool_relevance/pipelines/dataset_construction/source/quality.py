from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tool_relevance_lab.dataset_generation.schemas import CandidateSetRecord, JudgmentRecord, ToolUniverse


@dataclass(frozen=True)
class QualitySummary:
    samples: int
    plugin_exposures: dict[str, int]
    missing_agent_samples: list[str]
    invalid_judgement_samples: list[str]
    high_raw_score_ratio: float


def summarize_quality(
    *,
    universe: ToolUniverse,
    candidate_sets: list[CandidateSetRecord],
    judgments: list[JudgmentRecord],
    high_raw_score_threshold: float = 80.0,
) -> QualitySummary:
    agent_ids = {tool.tool_id for tool in universe.agents}
    plugin_ids = {tool.tool_id for tool in universe.plugins}
    exposures: Counter[str] = Counter()
    missing_agent_samples: list[str] = []

    for record in candidate_sets:
        tool_ids = set(record.candidate_set.tool_ids)
        exposures.update(tool_id for tool_id in tool_ids if tool_id in plugin_ids)
        if not agent_ids.issubset(tool_ids):
            missing_agent_samples.append(record.sample_id)

    invalid_judgement_samples: list[str] = []
    high_count = 0
    for judgment in judgments:
        ordered = set(judgment.candidate_tool_order)
        scored = {score.tool_id for score in judgment.scores}
        if ordered != scored:
            invalid_judgement_samples.append(judgment.sample_id)
        if judgment.scores and max(score.raw_score for score in judgment.scores) >= high_raw_score_threshold:
            high_count += 1

    high_ratio = high_count / len(judgments) if judgments else 0.0
    return QualitySummary(
        samples=len(candidate_sets),
        plugin_exposures=dict(exposures),
        missing_agent_samples=missing_agent_samples,
        invalid_judgement_samples=invalid_judgement_samples,
        high_raw_score_ratio=high_ratio,
    )
