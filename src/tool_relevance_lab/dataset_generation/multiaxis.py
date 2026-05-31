from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

from tool_relevance_lab.dataset_generation.jsonl import read_model_jsonl
from tool_relevance_lab.dataset_generation.judging import JudgeGenerationConfig, JudgeTask, build_judge_tasks
from tool_relevance_lab.dataset_generation.llm import LLMClient, LLMRequest
from tool_relevance_lab.dataset_generation.prompting import render_conversation, render_tool_list
from tool_relevance_lab.dataset_generation.resumable import ResumableRunSummary, run_resumable_jobs
from tool_relevance_lab.dataset_generation.schemas import (
    CandidateSetRecord,
    ConversationRecord,
    MultiAxisJudgmentRecord,
    MultiAxisTargetScore,
    MultiAxisToolScore,
    Provenance,
    RawAxisScore,
    ScoreAxis,
    ToolUniverse,
)


AXES: tuple[ScoreAxis, ...] = (
    "capability_match",
    "action_demand",
    "target_specificity",
    "consent_boundary",
    "intervention_cost",
    "companionship_fit",
    "final_preference",
)

FACTOR_AXES: tuple[ScoreAxis, ...] = (
    "capability_match",
    "action_demand",
    "target_specificity",
    "consent_boundary",
    "intervention_cost",
    "companionship_fit",
)


@dataclass(frozen=True)
class CalibratedMultiAxisToolScore:
    sample_id: str
    tool_id: str
    scores: dict[ScoreAxis, float]
    raw_scores: dict[ScoreAxis, float]
    local_z: dict[ScoreAxis, float]
    judge_disagreement: dict[ScoreAxis, float]

    def to_target(self) -> MultiAxisTargetScore:
        return MultiAxisTargetScore(
            tool_id=self.tool_id,
            scores=self.scores,
            raw_scores=self.raw_scores,
            local_z=self.local_z,
            judge_disagreement=self.judge_disagreement,
        )


async def judge_candidate_sets_multiaxis(
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

    async def worker(task: JudgeTask) -> MultiAxisJudgmentRecord:
        if dry_run:
            return build_dry_run_multiaxis_judgment(task=task, config=config)
        if client is None:
            raise RuntimeError("LLM client is required unless dry_run=True")
        request = LLMRequest(
            request_id=f"judge_multiaxis.{config.judge_run_id}.{task.sample_id}",
            idempotency_key=f"{config.prompt_version}:{config.judge_run_id}:multiaxis:{task.sample_id}",
            messages=build_multiaxis_judge_prompt(universe=universe, task=task, config=config),
            temperature=config.temperature,
            response_format={"type": "json_object"},
        )
        response = await client.complete(request)
        record = MultiAxisJudgmentRecord.model_validate(_json_object_from_text(response.text))
        return normalize_multiaxis_judgment_record(record, task=task, config=config)

    return await run_resumable_jobs(
        items=tasks,
        item_id=lambda task: task.sample_id,
        completed_id=lambda row: row.get("sample_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        concurrency=config.concurrency,
        max_attempts=config.max_attempts,
        progress_label="multiaxis_judgments",
    )


def build_multiaxis_judge_prompt(
    *,
    universe: ToolUniverse,
    task: JudgeTask,
    config: JudgeGenerationConfig,
) -> list[dict[str, str]]:
    candidate_ids = task.candidate_set.candidate_set.tool_ids
    extra = task.conversation.provenance.extra
    system = (
        "You are a multi-axis tool decision judge for N.E.K.O. Output strict JSON only. "
        "Use only each tool's source_text as capability evidence. Keep axes separate: "
        "do not copy the same positive or negative reason into every score."
    )
    user = (
        f"sample_id: {task.sample_id}\n"
        f"conversation_id: {task.conversation.conversation_id}\n"
        f"judge_run_id: {config.judge_run_id}\n\n"
        "Generator provenance hints (not labels; verify against the conversation):\n"
        f"- scenario_type: {extra.get('scenario_type', '')}\n"
        f"- tool_relevance_mode: {extra.get('tool_relevance_mode', '')}\n"
        f"- neko_context_type: {extra.get('neko_context_type', '')}\n"
        f"- authorization_level: {extra.get('authorization_level', '')}\n"
        f"- latest_user_actionability: {extra.get('latest_user_actionability', '')}\n\n"
        "N.E.K.O context:\n"
        "- N.E.K.O is emotional companionship plus optional agent/plugin abilities.\n"
        "- Tool recommendation is not pure semantic routing.\n"
        "- Topic overlap can be high while final recommendation stays low.\n"
        "- User boundaries and refusals are strong consent evidence, not capability evidence.\n"
        "- A tool can be capable and costly at the same time.\n\n"
        "Conversation:\n"
        f"{render_conversation(task.conversation)}\n\n"
        "Visible candidate tools in shuffled system order:\n"
        f"{render_tool_list(universe, candidate_ids)}\n\n"
        "Score every candidate on every axis from 0 to 100. Axis definitions:\n"
        "1. capability_match: If the user truly wanted the relevant task done, can this tool perform it? "
        "Ignore user intent, permission, risk, and tone.\n"
        "2. action_demand: Does the current conversation ask N.E.K.O to take action? "
        "This is mostly conversation-level. Do not make it high just because a tool is capable.\n"
        "3. target_specificity: Does the user's request point to this tool's capability category? "
        "This is pair-level. It differs from action_demand.\n"
        "4. consent_boundary: Is this tool intervention allowed by the current context? "
        "High means allowed or non-boundary; low means refused, sensitive, or needs confirmation not given.\n"
        "5. intervention_cost: How disruptive, proactive, privacy-sensitive, external, irreversible, "
        "or side-effectful would using/recommending this tool be here? High means costly/risky, not bad.\n"
        "6. companionship_fit: Would mentioning or using this tool feel natural for N.E.K.O's companionship "
        "moment? Ignore raw capability; judge timing, warmth, and whether it over-tools the moment.\n"
        "7. final_preference: Final recommendation strength after combining the factors.\n\n"
        "Orthogonality rules:\n"
        "- capability_match high does not imply action_demand, consent_boundary, companionship_fit, or final high.\n"
        "- action_demand is not tool capability. For a given conversation it should usually vary little across tools.\n"
        "- target_specificity can be high even if consent_boundary is low.\n"
        "- intervention_cost can be high for a useful tool. It is not an inverse final score.\n"
        "- companionship_fit can be low for a capable tool in a tender/no-tool moment.\n"
        "- If consent_boundary is very low, final_preference must be low even with high capability_match.\n"
        "- High intervention_cost needs strong action_demand and consent_boundary to support high final_preference.\n"
        "- Keep scores spread when evidence differs. Avoid putting all axes in the same narrow band.\n\n"
        "Return strict JSON with exactly this shape:\n"
        "{"
        "\"sample_id\":\"...\","
        "\"conversation_id\":\"...\","
        "\"judge_run_id\":\"...\","
        "\"candidate_tool_order\":[\"...\"],"
        "\"scores\":[{"
        "\"tool_id\":\"...\","
        "\"scores\":["
        "{\"axis\":\"capability_match\",\"raw_score\":0.0},"
        "{\"axis\":\"action_demand\",\"raw_score\":0.0},"
        "{\"axis\":\"target_specificity\",\"raw_score\":0.0},"
        "{\"axis\":\"consent_boundary\",\"raw_score\":0.0},"
        "{\"axis\":\"intervention_cost\",\"raw_score\":0.0},"
        "{\"axis\":\"companionship_fit\",\"raw_score\":0.0},"
        "{\"axis\":\"final_preference\",\"raw_score\":0.0}"
        "],"
        "\"rationale\":{\"capability_match\":\"short reason\",\"final_preference\":\"short reason\"}"
        "}],"
        "\"provenance\":{\"model\":\"...\",\"prompt_version\":\"neko_tool_multiaxis_judge_v1\","
        "\"seed\":null,\"extra\":{}}"
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_multiaxis_judgment_record(
    record: MultiAxisJudgmentRecord,
    *,
    task: JudgeTask,
    config: JudgeGenerationConfig,
) -> MultiAxisJudgmentRecord:
    candidate_order = task.candidate_set.candidate_set.tool_ids
    scores_by_id = {score.tool_id: score for score in record.scores}
    missing = [tool_id for tool_id in candidate_order if tool_id not in scores_by_id]
    extra = [tool_id for tool_id in scores_by_id if tool_id not in set(candidate_order)]
    if missing or extra:
        raise ValueError(f"judgment scores mismatch for sample_id={task.sample_id}; missing={missing}; extra={extra}")
    normalized_scores = [
        MultiAxisToolScore(
            tool_id=tool_id,
            scores=_normalize_axis_scores(scores_by_id[tool_id].scores),
            rationale=scores_by_id[tool_id].rationale,
        )
        for tool_id in candidate_order
    ]
    return MultiAxisJudgmentRecord(
        sample_id=task.sample_id,
        conversation_id=task.conversation.conversation_id,
        judge_run_id=config.judge_run_id,
        candidate_tool_order=candidate_order,
        scores=normalized_scores,
        provenance=Provenance(
            model=config.model,
            prompt_version=config.prompt_version,
            extra=record.provenance.extra,
        ),
    )


def build_dry_run_multiaxis_judgment(*, task: JudgeTask, config: JudgeGenerationConfig) -> MultiAxisJudgmentRecord:
    candidate_order = task.candidate_set.candidate_set.tool_ids
    targets = set(task.conversation.generation_hint.target_tool_ids)
    mode = str(task.conversation.provenance.extra.get("tool_relevance_mode", "actionable_tool_relevance"))
    no_tool = mode == "no_tool_or_low_relevance"
    weak = mode == "weak_or_requires_confirmation"
    scores = []
    for index, tool_id in enumerate(candidate_order):
        is_target = tool_id in targets
        action = 25.0 if no_tool else 55.0 if weak else 88.0
        capability = 88.0 if is_target else max(8.0, 35.0 - index)
        target_specificity = 82.0 if is_target else max(5.0, 30.0 - index)
        consent = 35.0 if no_tool else 58.0 if weak else 82.0
        cost = 72.0 if tool_id.startswith("agent.") else 42.0
        style = 35.0 if no_tool and is_target else 62.0 if weak else 76.0
        final = 20.0 if no_tool and is_target else 52.0 if weak and is_target else 90.0 if is_target else 12.0
        axis_values = {
            "capability_match": capability,
            "action_demand": action,
            "target_specificity": target_specificity,
            "consent_boundary": consent,
            "intervention_cost": cost,
            "companionship_fit": style,
            "final_preference": final,
        }
        scores.append(
            MultiAxisToolScore(
                tool_id=tool_id,
                scores=[RawAxisScore(axis=axis, raw_score=value) for axis, value in axis_values.items()],
                rationale={"dry_run": "heuristic multiaxis judgment"},
            )
        )
    return MultiAxisJudgmentRecord(
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


def calibrate_multiaxis_confidences(
    judgments: list[MultiAxisJudgmentRecord],
    *,
    clip_percentile: float = 0.001,
) -> list[CalibratedMultiAxisToolScore]:
    if not 0.0 <= clip_percentile < 0.5:
        raise ValueError("clip_percentile must be in [0, 0.5)")

    aggregated = aggregate_multiaxis_judgments(judgments)
    raw_by_key_axis = {(sample_id, tool_id, axis): values for (sample_id, tool_id, axis), values in aggregated.items()}
    sample_tool_keys = sorted({(sample_id, tool_id) for sample_id, tool_id, _ in aggregated})
    calibrated_by_axis: dict[ScoreAxis, dict[tuple[str, str], tuple[float, float, float]]] = {}
    for axis in AXES:
        calibrated_by_axis[axis] = _calibrate_axis(raw_by_key_axis, axis=axis, clip_percentile=clip_percentile)

    rows = []
    for sample_id, tool_id in sample_tool_keys:
        scores = {}
        raw_scores = {}
        local_z = {}
        disagreement = {}
        for axis in AXES:
            raw, z_value, calibrated, axis_disagreement = calibrated_by_axis[axis][(sample_id, tool_id)]
            scores[axis] = calibrated
            raw_scores[axis] = raw
            local_z[axis] = z_value
            disagreement[axis] = axis_disagreement
        rows.append(
            CalibratedMultiAxisToolScore(
                sample_id=sample_id,
                tool_id=tool_id,
                scores=scores,
                raw_scores=raw_scores,
                local_z=local_z,
                judge_disagreement=disagreement,
            )
        )
    return rows


def aggregate_multiaxis_judgments(
    judgments: list[MultiAxisJudgmentRecord],
) -> dict[tuple[str, str, ScoreAxis], list[float]]:
    grouped: dict[tuple[str, str, ScoreAxis], list[float]] = defaultdict(list)
    for judgment in judgments:
        for tool_score in judgment.scores:
            axis_scores = {score.axis: score.raw_score for score in tool_score.scores}
            missing = [axis for axis in AXES if axis not in axis_scores]
            if missing:
                raise ValueError(f"missing axes for {judgment.sample_id}/{tool_score.tool_id}: {missing}")
            for axis in AXES:
                grouped[(judgment.sample_id, tool_score.tool_id, axis)].append(axis_scores[axis])
    return grouped


def _calibrate_axis(
    raw_by_key_axis: dict[tuple[str, str, ScoreAxis], list[float]],
    *,
    axis: ScoreAxis,
    clip_percentile: float,
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    by_sample: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for (sample_id, tool_id, item_axis), values in raw_by_key_axis.items():
        if item_axis != axis:
            continue
        raw = float(statistics.median(values))
        disagreement = float(max(values) - min(values))
        by_sample[sample_id].append((tool_id, raw, disagreement))

    local_rows: list[tuple[str, str, float, float, float]] = []
    for sample_id, rows in by_sample.items():
        raw_scores = [raw for _, raw, _ in rows]
        mean = statistics.fmean(raw_scores)
        std = statistics.pstdev(raw_scores)
        if std == 0.0:
            local_z_values = [0.0 for _ in raw_scores]
        else:
            local_z_values = [(raw - mean) / std for raw in raw_scores]
        for (tool_id, raw, disagreement), z_value in zip(rows, local_z_values, strict=True):
            local_rows.append((sample_id, tool_id, raw, z_value, disagreement))

    calibrated = _rank_gaussian([row[3] for row in local_rows], clip_percentile=clip_percentile)
    return {
        (sample_id, tool_id): (raw, z_value, score, disagreement)
        for (sample_id, tool_id, raw, z_value, disagreement), score in zip(local_rows, calibrated, strict=True)
    }


def _normalize_axis_scores(scores: list[RawAxisScore]) -> list[RawAxisScore]:
    by_axis = {score.axis: float(score.raw_score) for score in scores}
    missing = [axis for axis in AXES if axis not in by_axis]
    if missing:
        raise ValueError(f"missing axis scores: {missing}")
    return [RawAxisScore(axis=axis, raw_score=by_axis[axis]) for axis in AXES]


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


def multiaxis_rows_to_jsonl(rows: list[CalibratedMultiAxisToolScore]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row.sample_id,
            "tool_id": row.tool_id,
            "confidence": row.scores["final_preference"],
            "raw_score": row.raw_scores["final_preference"],
            "axis_scores": row.scores,
            "raw_scores": row.raw_scores,
            "local_z": row.local_z,
            "judge_disagreement": row.judge_disagreement,
        }
        for row in rows
    ]


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
