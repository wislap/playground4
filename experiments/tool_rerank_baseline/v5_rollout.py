from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from tool_relevance_lab.dataset_generation.llm import LLMClient, LLMRequest
from tool_relevance_lab.dataset_generation.resumable import ResumableRunSummary, run_resumable_jobs
from v5_action_policy import ActionRecommendation, DesiredStateGapPolicy, ToolState


SlateKind = Literal["anchor", "conservative", "sampled", "over_recommend", "hard_negative"]


@dataclass(frozen=True)
class V5RolloutState:
    state_id: str
    conversation_excerpt: str
    tools: list[ToolState]


@dataclass(frozen=True)
class V5Slate:
    slate_id: str
    kind: SlateKind
    actions: list[ActionRecommendation]
    selected_mask: list[bool]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "slate_id": self.slate_id,
            "kind": self.kind,
            "actions": [
                {
                    "tool_id": action.tool_id,
                    "action": action.action,
                    "gain": round(action.gain, 4),
                    "desired_probability": round(action.desired_probability, 4),
                }
                for action in self.actions
            ],
        }


@dataclass(frozen=True)
class V5JudgedAction:
    tool_id: str
    action: str
    score: float


@dataclass(frozen=True)
class V5JudgedSlate:
    slate_id: str
    slate_score: float
    action_scores: list[V5JudgedAction]


@dataclass(frozen=True)
class V5JudgeRecord:
    state_id: str
    judged_slates: list[V5JudgedSlate]
    rationale: str = ""


def sample_v5_slates(
    *,
    policy: DesiredStateGapPolicy,
    state: V5RolloutState,
    group_size: int = 5,
    max_actions_per_slate: int = 5,
    seed: int = 0,
) -> list[V5Slate]:
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if max_actions_per_slate < 1:
        raise ValueError("max_actions_per_slate must be >= 1")
    if not state.tools:
        return []

    rng = random.Random(seed)
    scores = torch.tensor([tool.score for tool in state.tools], dtype=torch.float32)
    enabled = torch.tensor([tool.enabled for tool in state.tools], dtype=torch.bool)
    with torch.no_grad():
        output = policy(scores, enabled)
    desired = output["desired_probability"].cpu().tolist()
    gains = output["gain"].cpu().tolist()
    probabilities = output["action_probability"].cpu().tolist()
    thresholds = output["threshold"].cpu().tolist()

    candidates = [
        _candidate_action(tool=tool, desired=desired[index], gain=gains[index], probability=probabilities[index])
        for index, tool in enumerate(state.tools)
    ]
    high_to_low = sorted(range(len(state.tools)), key=lambda index: gains[index], reverse=True)

    slate_specs: list[tuple[SlateKind, list[int]]] = [
        ("anchor", [index for index in high_to_low if gains[index] > thresholds[index]]),
        ("conservative", [index for index in high_to_low if gains[index] > min(0.98, thresholds[index] + 0.10)]),
        ("sampled", [index for index in high_to_low if rng.random() < probabilities[index]]),
        ("over_recommend", [index for index in high_to_low if gains[index] > max(0.0, thresholds[index] - 0.18)]),
        ("hard_negative", _hard_negative_indices(high_to_low=high_to_low, gains=gains, thresholds=thresholds)),
    ]
    while len(slate_specs) < group_size:
        sampled = [index for index in high_to_low if rng.random() < probabilities[index]]
        slate_specs.append(("sampled", sampled))

    slates: list[V5Slate] = []
    for slate_index, (kind, selected_indices) in enumerate(slate_specs[:group_size], start=1):
        selected_indices = _dedupe_preserve_order(selected_indices)[:max_actions_per_slate]
        selected = set(selected_indices)
        actions = [candidates[index] for index in selected_indices]
        slates.append(
            V5Slate(
                slate_id=f"{state.state_id}.slate_{slate_index:02d}",
                kind=kind,
                actions=actions,
                selected_mask=[index in selected for index in range(len(state.tools))],
            )
        )
    return slates


def build_v5_judge_prompt(
    *,
    state: V5RolloutState,
    slates: list[V5Slate],
    user_persona: str = "A sensitive N.E.K.O user who values warmth, low interruption, and consent.",
) -> list[dict[str, str]]:
    system = (
        "You judge plugin-state recommendations for N.E.K.O, an emotional companion with optional "
        "background tools. Reward recommendations that feel helpful, respectful, low-interruption, "
        "and aligned with the current conversation. Empty recommendations can be best when no state "
        "change is needed. Do not reward doing more merely because more actions are present."
    )
    user = {
        "state_id": state.state_id,
        "user_persona": user_persona,
        "conversation_excerpt": state.conversation_excerpt,
        "scoring_scale": {
            "slate_score": "-1.0 bad / annoying / wrong, 0 neutral, 1.0 excellent",
            "action_score": "-1.0 wrong action, 0 neutral, 1.0 clearly right action",
        },
        "tools": [
            {
                "tool_id": tool.tool_id,
                "v22b_score": round(tool.score, 4),
                "enabled": tool.enabled,
            }
            for tool in state.tools
        ],
        "candidate_slates": [slate.to_prompt_dict() for slate in slates],
        "return_json_shape": {
            "state_id": state.state_id,
            "slates": [
                {
                    "slate_id": "same as candidate",
                    "slate_score": 0.0,
                    "action_scores": [
                        {"tool_id": "tool id", "action": "enable|disable", "score": 0.0}
                    ],
                }
            ],
            "rationale": "short explanation",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2, sort_keys=True)},
    ]


def parse_v5_judge_response(text: str, *, state_id: str, expected_slate_ids: set[str]) -> V5JudgeRecord:
    payload = _json_object_from_text(text)
    if str(payload.get("state_id")) != state_id:
        raise ValueError(f"judge response state_id mismatch: expected={state_id}")
    judged_slates: list[V5JudgedSlate] = []
    seen: set[str] = set()
    for row in payload.get("slates", []):
        slate_id = str(row.get("slate_id", ""))
        if slate_id not in expected_slate_ids:
            raise ValueError(f"unexpected slate_id in judge response: {slate_id}")
        seen.add(slate_id)
        judged_slates.append(
            V5JudgedSlate(
                slate_id=slate_id,
                slate_score=float(row.get("slate_score", 0.0)),
                action_scores=[
                    V5JudgedAction(
                        tool_id=str(action.get("tool_id", "")),
                        action=str(action.get("action", "")),
                        score=float(action.get("score", 0.0)),
                    )
                    for action in row.get("action_scores", [])
                ],
            )
        )
    missing = expected_slate_ids - seen
    if missing:
        raise ValueError(f"judge response missing slate ids: {sorted(missing)}")
    return V5JudgeRecord(
        state_id=state_id,
        judged_slates=judged_slates,
        rationale=str(payload.get("rationale", "")),
    )


def rollout_record_to_dict(
    *,
    state: V5RolloutState,
    slates: list[V5Slate],
    judge: V5JudgeRecord,
) -> dict[str, Any]:
    return {
        "state": {
            "state_id": state.state_id,
            "conversation_excerpt": state.conversation_excerpt,
            "tools": [asdict(tool) for tool in state.tools],
        },
        "slates": [
            {
                "slate_id": slate.slate_id,
                "kind": slate.kind,
                "selected_mask": slate.selected_mask,
                "actions": [asdict(action) for action in slate.actions],
            }
            for slate in slates
        ],
        "judge": {
            "state_id": judge.state_id,
            "rationale": judge.rationale,
            "slates": [
                {
                    "slate_id": slate.slate_id,
                    "slate_score": slate.slate_score,
                    "action_scores": [asdict(action) for action in slate.action_scores],
                }
                for slate in judge.judged_slates
            ],
        },
    }


async def judge_v5_rollout_states(
    *,
    policy: DesiredStateGapPolicy,
    states: list[V5RolloutState],
    client: LLMClient | None,
    output_path,
    error_path,
    model: str,
    prompt_version: str = "neko_v5_action_slate_judge_v1",
    temperature: float = 0.4,
    concurrency: int = 8,
    max_attempts: int = 3,
    group_size: int = 5,
    max_actions_per_slate: int = 5,
    seed: int = 20260601,
    dry_run: bool = False,
) -> ResumableRunSummary:
    async def worker(state: V5RolloutState) -> dict[str, Any]:
        slates = sample_v5_slates(
            policy=policy,
            state=state,
            group_size=group_size,
            max_actions_per_slate=max_actions_per_slate,
            seed=seed + _stable_int(state.state_id),
        )
        if dry_run:
            judge = _dry_run_judge(state=state, slates=slates)
        else:
            if client is None:
                raise RuntimeError("LLM client is required unless dry_run=True")
            request = LLMRequest(
                request_id=f"v5_judge.{state.state_id}",
                idempotency_key=f"{prompt_version}:{model}:{state.state_id}",
                messages=build_v5_judge_prompt(state=state, slates=slates),
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            response = await client.complete(request)
            judge = parse_v5_judge_response(
                response.text,
                state_id=state.state_id,
                expected_slate_ids={slate.slate_id for slate in slates},
            )
        row = rollout_record_to_dict(state=state, slates=slates, judge=judge)
        row["provenance"] = {
            "model": model,
            "prompt_version": prompt_version,
            "dry_run": dry_run,
        }
        return row

    return await run_resumable_jobs(
        items=states,
        item_id=lambda state: state.state_id,
        completed_id=lambda row: row.get("state", {}).get("state_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        concurrency=concurrency,
        max_attempts=max_attempts,
        progress_label="v5_rollout_judge",
    )


def _candidate_action(
    *,
    tool: ToolState,
    desired: float,
    gain: float,
    probability: float,
) -> ActionRecommendation:
    return ActionRecommendation(
        tool_id=tool.tool_id,
        action="disable" if tool.enabled else "enable",
        desired_probability=float(desired),
        gain=float(gain),
        action_probability=float(probability),
    )


def _dry_run_judge(*, state: V5RolloutState, slates: list[V5Slate]) -> V5JudgeRecord:
    judged = []
    for slate in slates:
        action_scores = []
        slate_score = 0.0
        for action in slate.actions:
            score = 0.8 if action.gain >= 0.75 else -0.4
            action_scores.append(
                V5JudgedAction(tool_id=action.tool_id, action=action.action, score=score)
            )
            slate_score += score
        if not slate.actions:
            slate_score = 0.2
        else:
            slate_score = slate_score / len(slate.actions)
            slate_score -= max(0, len(slate.actions) - 2) * 0.15
        judged.append(
            V5JudgedSlate(
                slate_id=slate.slate_id,
                slate_score=float(max(-1.0, min(1.0, slate_score))),
                action_scores=action_scores,
            )
        )
    return V5JudgeRecord(
        state_id=state.state_id,
        judged_slates=judged,
        rationale="dry-run heuristic: high-gain sparse slates are preferred.",
    )


def _hard_negative_indices(
    *,
    high_to_low: list[int],
    gains: list[float],
    thresholds: list[float],
) -> list[int]:
    strong = [index for index in high_to_low if gains[index] > thresholds[index]]
    weak = [index for index in reversed(high_to_low) if gains[index] <= thresholds[index]]
    if strong and weak:
        return [*strong[:-1], weak[0]]
    if weak:
        return [weak[0]]
    return high_to_low[-1:] if high_to_low else []


def _dedupe_preserve_order(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        output.append(index)
    return output


def _stable_int(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 131 + ord(char)) % 1_000_000_007
    return total


def _json_object_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])
