from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist

from tool_relevance_lab.dataset_generation.schemas import JudgmentRecord


@dataclass(frozen=True)
class CalibratedToolScore:
    sample_id: str
    tool_id: str
    raw_score: float
    local_z: float
    confidence: float
    judge_disagreement: float


def aggregate_judgments(judgments: list[JudgmentRecord]) -> dict[tuple[str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for judgment in judgments:
        for score in judgment.scores:
            grouped[(judgment.sample_id, score.tool_id)].append(score.raw_score)

    aggregated: dict[tuple[str, str], tuple[float, float]] = {}
    for key, values in grouped.items():
        aggregated[key] = (float(statistics.median(values)), float(max(values) - min(values)))
    return aggregated


def calibrate_confidences(
    judgments: list[JudgmentRecord],
    *,
    clip_percentile: float = 0.001,
) -> list[CalibratedToolScore]:
    if not 0.0 <= clip_percentile < 0.5:
        raise ValueError("clip_percentile must be in [0, 0.5)")

    aggregated = aggregate_judgments(judgments)
    by_sample: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for (sample_id, tool_id), (raw_score, disagreement) in aggregated.items():
        by_sample[sample_id].append((tool_id, raw_score, disagreement))

    local_rows: list[tuple[str, str, float, float, float]] = []
    for sample_id, rows in by_sample.items():
        raw_scores = [raw_score for _, raw_score, _ in rows]
        mean = statistics.fmean(raw_scores)
        std = statistics.pstdev(raw_scores)
        if std == 0.0:
            local_z_values = [0.0 for _ in raw_scores]
        else:
            local_z_values = [(raw_score - mean) / std for raw_score in raw_scores]
        for (tool_id, raw_score, disagreement), local_z in zip(rows, local_z_values, strict=True):
            local_rows.append((sample_id, tool_id, raw_score, local_z, disagreement))

    ranked = _rank_gaussian([row[3] for row in local_rows], clip_percentile=clip_percentile)
    return [
        CalibratedToolScore(
            sample_id=sample_id,
            tool_id=tool_id,
            raw_score=raw_score,
            local_z=local_z,
            confidence=confidence,
            judge_disagreement=disagreement,
        )
        for (sample_id, tool_id, raw_score, local_z, disagreement), confidence in zip(
            local_rows, ranked, strict=True
        )
    ]


def _rank_gaussian(values: list[float], *, clip_percentile: float) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end

    normal = NormalDist()
    total = len(values)
    calibrated: list[float] = []
    for rank in ranks:
        percentile = (rank - 0.5) / total
        percentile = min(max(percentile, clip_percentile), 1.0 - clip_percentile)
        calibrated.append(normal.inv_cdf(percentile))
    return calibrated
