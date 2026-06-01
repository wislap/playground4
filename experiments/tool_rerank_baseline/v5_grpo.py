from __future__ import annotations

from dataclasses import dataclass

import torch

from v5_action_policy import DesiredStateGapPolicy, ToolState
from v5_rollout import V5JudgeRecord, V5Slate


@dataclass(frozen=True)
class V5GroupLoss:
    loss: torch.Tensor
    slate_loss: torch.Tensor
    action_loss: torch.Tensor
    mean_reward: float


def compute_v5_multiscale_grpo_loss(
    *,
    policy: DesiredStateGapPolicy,
    tools: list[ToolState],
    slates: list[V5Slate],
    judge: V5JudgeRecord,
    action_weight: float = 0.35,
    entropy_weight: float = 0.0,
) -> V5GroupLoss:
    if not tools:
        raise ValueError("tools must not be empty")
    if not slates:
        raise ValueError("slates must not be empty")
    if action_weight < 0:
        raise ValueError("action_weight must be non-negative")
    if entropy_weight < 0:
        raise ValueError("entropy_weight must be non-negative")

    slate_by_id = {slate.slate_id: slate for slate in slates}
    judge_by_id = {slate.slate_id: slate for slate in judge.judged_slates}
    missing = set(slate_by_id) - set(judge_by_id)
    if missing:
        raise ValueError(f"judge record missing slate ids: {sorted(missing)}")

    scores = torch.tensor([tool.score for tool in tools], dtype=torch.float32)
    enabled = torch.tensor([tool.enabled for tool in tools], dtype=torch.bool)
    selected = torch.tensor([slate.selected_mask for slate in slates], dtype=torch.bool)
    per_tool_log_prob = policy.log_prob(
        scores=scores.unsqueeze(0).expand(len(slates), -1).reshape(-1),
        enabled=enabled.unsqueeze(0).expand(len(slates), -1).reshape(-1),
        selected=selected.reshape(-1),
    ).reshape(len(slates), len(tools))

    slate_log_prob = per_tool_log_prob.sum(dim=1)
    slate_rewards = torch.tensor(
        [float(judge_by_id[slate.slate_id].slate_score) for slate in slates],
        dtype=torch.float32,
    )
    slate_advantage = _normalize(slate_rewards)
    slate_loss = -(slate_advantage.detach() * slate_log_prob).mean()

    action_terms: list[torch.Tensor] = []
    for slate_index, slate in enumerate(slates):
        judged_actions = judge_by_id[slate.slate_id].action_scores
        if not judged_actions:
            continue
        action_rewards = torch.tensor([action.score for action in judged_actions], dtype=torch.float32)
        action_advantage = _normalize(action_rewards)
        selected_indices = _action_indices(tools=tools, slate=slate, judged_actions=judged_actions)
        for local_index, tool_index in enumerate(selected_indices):
            action_terms.append(
                -action_advantage[local_index].detach() * per_tool_log_prob[slate_index, tool_index]
            )
    if action_terms:
        action_loss = torch.stack(action_terms).mean()
    else:
        action_loss = torch.zeros((), dtype=torch.float32)

    entropy_penalty = torch.zeros((), dtype=torch.float32)
    if entropy_weight:
        probability = policy(scores, enabled)["action_probability"]
        entropy = -(
            probability * torch.log(probability.clamp_min(1e-8))
            + (1.0 - probability) * torch.log((1.0 - probability).clamp_min(1e-8))
        ).mean()
        entropy_penalty = -entropy_weight * entropy

    loss = slate_loss + action_weight * action_loss + entropy_penalty
    return V5GroupLoss(
        loss=loss,
        slate_loss=slate_loss,
        action_loss=action_loss,
        mean_reward=float(slate_rewards.mean()),
    )


def _normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean()
    std = values.std(unbiased=False)
    if float(std) <= 1e-6:
        return torch.zeros_like(values)
    return centered / std.clamp_min(1e-6)




def _action_indices(*, tools: list[ToolState], slate: V5Slate, judged_actions) -> list[int]:
    index_by_action = {
        (action.tool_id, action.action): index
        for index, action in enumerate(slate.actions)
    }
    tool_index_by_id = {tool.tool_id: index for index, tool in enumerate(tools)}
    indices: list[int] = []
    for judged_action in judged_actions:
        slate_action_index = index_by_action.get((judged_action.tool_id, judged_action.action))
        if slate_action_index is None:
            raise ValueError(
                f"judge action not present in slate {slate.slate_id}: "
                f"{judged_action.action} {judged_action.tool_id}"
            )
        indices.append(tool_index_by_id[judged_action.tool_id])
    return indices
