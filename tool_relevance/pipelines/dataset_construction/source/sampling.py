from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from tool_relevance_lab.dataset_generation.schemas import (
    CandidateSet,
    CandidateSetRecord,
    ToolUniverse,
)


@dataclass
class CandidateSamplingConfig:
    plugin_sample_rate: float = 0.10
    target_force_rate: float = 0.80
    weight_decay_on_select: float = 0.70
    shuffle_candidates: bool = True


@dataclass
class CandidateSampler:
    universe: ToolUniverse
    config: CandidateSamplingConfig = field(default_factory=CandidateSamplingConfig)
    weights: dict[str, float] = field(init=False)
    exposure_counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.config.plugin_sample_rate <= 1.0:
            raise ValueError("plugin_sample_rate must be in [0, 1]")
        if not 0.0 <= self.config.target_force_rate <= 1.0:
            raise ValueError("target_force_rate must be in [0, 1]")
        if not 0.0 < self.config.weight_decay_on_select <= 1.0:
            raise ValueError("weight_decay_on_select must be in (0, 1]")
        self.weights = {tool.tool_id: 1.0 for tool in self.universe.plugins}
        self.exposure_counts = {tool.tool_id: 0 for tool in self.universe.plugins}

    def sample(
        self,
        *,
        sample_id: str,
        conversation_id: str,
        seed: int,
        target_tool_ids: list[str] | None = None,
    ) -> CandidateSetRecord:
        rng = random.Random(seed)
        plugin_ids = [tool.tool_id for tool in self.universe.plugins]
        agent_ids = [tool.tool_id for tool in self.universe.agents]
        k = math.ceil(len(plugin_ids) * self.config.plugin_sample_rate)
        forced = self._forced_targets(rng, target_tool_ids or [], plugin_ids, k)
        selected_plugins = list(forced)
        fill_count = max(0, k - len(selected_plugins))
        available = [tool_id for tool_id in plugin_ids if tool_id not in forced]
        selected_plugins.extend(self._weighted_sample_without_replacement(rng, available, fill_count))

        self._update_weights(selected_plugins, plugin_ids)

        tool_ids = agent_ids + selected_plugins
        if self.config.shuffle_candidates:
            rng.shuffle(tool_ids)

        return CandidateSetRecord(
            sample_id=sample_id,
            conversation_id=conversation_id,
            candidate_set=CandidateSet(
                sampling_seed=seed,
                plugin_sample_rate=self.config.plugin_sample_rate,
                tool_ids=tool_ids,
            ),
        )

    def _forced_targets(
        self,
        rng: random.Random,
        target_tool_ids: list[str],
        plugin_ids: list[str],
        max_count: int,
    ) -> list[str]:
        forced: list[str] = []
        for tool_id in target_tool_ids:
            if tool_id not in plugin_ids:
                continue
            if len(forced) >= max_count:
                break
            if rng.random() <= self.config.target_force_rate:
                forced.append(tool_id)
        return list(dict.fromkeys(forced))

    def _weighted_sample_without_replacement(
        self,
        rng: random.Random,
        plugin_ids: list[str],
        count: int,
    ) -> list[str]:
        selected: list[str] = []
        remaining = list(plugin_ids)
        for _ in range(min(count, len(remaining))):
            total = sum(max(self.weights[tool_id], 0.0) for tool_id in remaining)
            if total <= 0.0:
                choice = rng.choice(remaining)
            else:
                threshold = rng.random() * total
                cumulative = 0.0
                choice = remaining[-1]
                for tool_id in remaining:
                    cumulative += max(self.weights[tool_id], 0.0)
                    if cumulative >= threshold:
                        choice = tool_id
                        break
            selected.append(choice)
            remaining.remove(choice)
        return selected

    def _update_weights(self, selected: list[str], all_plugin_ids: list[str]) -> None:
        selected_set = set(selected)
        removed_mass = 0.0
        for tool_id in selected_set:
            self.exposure_counts[tool_id] += 1
            old = self.weights[tool_id]
            self.weights[tool_id] *= self.config.weight_decay_on_select
            removed_mass += old - self.weights[tool_id]

        unselected = [tool_id for tool_id in all_plugin_ids if tool_id not in selected_set]
        if not unselected or removed_mass <= 0.0:
            return
        bonus = removed_mass / len(unselected)
        for tool_id in unselected:
            self.weights[tool_id] += bonus
