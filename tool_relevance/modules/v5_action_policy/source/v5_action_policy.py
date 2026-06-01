from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn


ActionType = Literal["enable", "disable"]


@dataclass(frozen=True)
class ToolState:
    tool_id: str
    score: float
    enabled: bool


@dataclass(frozen=True)
class ActionRecommendation:
    tool_id: str
    action: ActionType
    desired_probability: float
    gain: float
    action_probability: float


@dataclass(frozen=True)
class V5PolicySnapshot:
    mu: float
    temperature: float
    enable_threshold: float
    disable_threshold: float
    action_temperature: float


class DesiredStateGapPolicy(nn.Module):
    """Lightweight V5 policy that maps V2.2b fit scores to plugin state changes.

    V2.2b owns semantic relevance. This module only learns how aggressively a
    score should move current plugin state toward the implied desired state.
    """

    def __init__(
        self,
        *,
        mu: float = 0.0,
        temperature: float = 1.0,
        enable_threshold: float = 0.80,
        disable_threshold: float = 0.80,
        action_temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if action_temperature <= 0:
            raise ValueError("action_temperature must be positive")
        _validate_probability("enable_threshold", enable_threshold)
        _validate_probability("disable_threshold", disable_threshold)

        self.mu = nn.Parameter(torch.tensor(float(mu), dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.enable_threshold_logit = nn.Parameter(_logit_tensor(enable_threshold))
        self.disable_threshold_logit = nn.Parameter(_logit_tensor(disable_threshold))
        self.log_action_temperature = nn.Parameter(torch.log(torch.tensor(float(action_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(1e-4)

    @property
    def enable_threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.enable_threshold_logit)

    @property
    def disable_threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.disable_threshold_logit)

    @property
    def action_temperature(self) -> torch.Tensor:
        return self.log_action_temperature.exp().clamp_min(1e-4)

    def snapshot(self) -> V5PolicySnapshot:
        return V5PolicySnapshot(
            mu=float(self.mu.detach().cpu()),
            temperature=float(self.temperature.detach().cpu()),
            enable_threshold=float(self.enable_threshold.detach().cpu()),
            disable_threshold=float(self.disable_threshold.detach().cpu()),
            action_temperature=float(self.action_temperature.detach().cpu()),
        )

    def desired_probability(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((scores - self.mu) / self.temperature)

    def forward(self, scores: torch.Tensor, enabled: torch.Tensor) -> dict[str, torch.Tensor]:
        enabled_bool = enabled.bool()
        desired = self.desired_probability(scores.float())
        enable_gain = desired
        disable_gain = 1.0 - desired
        gain = torch.where(enabled_bool, disable_gain, enable_gain)
        threshold = torch.where(
            enabled_bool,
            self.disable_threshold.expand_as(gain),
            self.enable_threshold.expand_as(gain),
        )
        action_probability = torch.sigmoid((gain - threshold) / self.action_temperature)
        signed_gain = torch.where(enabled_bool, -disable_gain, enable_gain)
        signed_action_probability = torch.where(
            enabled_bool,
            -action_probability,
            action_probability,
        )
        return {
            "desired_probability": desired,
            "gain": gain,
            "signed_gain": signed_gain,
            "action_probability": action_probability,
            "signed_action_probability": signed_action_probability,
            "threshold": threshold,
        }

    def log_prob(
        self,
        *,
        scores: torch.Tensor,
        enabled: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        """Bernoulli log-probability for action-level DPO/GRPO training."""
        probability = self(scores, enabled)["action_probability"]
        selected_float = selected.float()
        return (
            selected_float * torch.log(probability.clamp_min(1e-8))
            + (1.0 - selected_float) * torch.log((1.0 - probability).clamp_min(1e-8))
        )

    @torch.no_grad()
    def recommend(self, tools: list[ToolState], *, max_actions: int | None = None) -> list[ActionRecommendation]:
        if max_actions is not None and max_actions < 1:
            raise ValueError("max_actions must be positive when provided")
        if not tools:
            return []

        scores = torch.tensor([tool.score for tool in tools], dtype=torch.float32)
        enabled = torch.tensor([tool.enabled for tool in tools], dtype=torch.bool)
        output = self(scores, enabled)
        desired = output["desired_probability"].cpu().numpy()
        gains = output["gain"].cpu().numpy()
        probabilities = output["action_probability"].cpu().numpy()
        thresholds = output["threshold"].cpu().numpy()

        actions: list[ActionRecommendation] = []
        for index, tool in enumerate(tools):
            if gains[index] <= thresholds[index]:
                continue
            actions.append(
                ActionRecommendation(
                    tool_id=tool.tool_id,
                    action="disable" if tool.enabled else "enable",
                    desired_probability=float(desired[index]),
                    gain=float(gains[index]),
                    action_probability=float(probabilities[index]),
                )
            )

        actions.sort(key=lambda action: action.gain, reverse=True)
        if max_actions is not None:
            return actions[:max_actions]
        return actions


def action_feedback_advantages(
    actions: list[ActionRecommendation],
    ratings: list[float],
) -> dict[str, float]:
    """Convert action-level relative ratings into score-direction advantages.

    Positive values mean "raise this tool's V2.2b score in similar contexts";
    negative values mean "lower it". Disable actions invert the direction.
    """
    if len(actions) != len(ratings):
        raise ValueError("actions and ratings must have the same length")
    if not actions:
        return {}
    rating_array = np.asarray(ratings, dtype="float32")
    centered = rating_array - float(rating_array.mean())
    std = float(rating_array.std())
    if std > 1e-6:
        centered = centered / std

    advantages: dict[str, float] = {}
    for action, advantage in zip(actions, centered, strict=True):
        direction = 1.0 if action.action == "enable" else -1.0
        advantages[action.tool_id] = float(direction * advantage)
    return advantages


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")


def _logit_tensor(value: float) -> torch.Tensor:
    _validate_probability("value", value)
    tensor = torch.tensor(float(value), dtype=torch.float32)
    return torch.logit(tensor)
