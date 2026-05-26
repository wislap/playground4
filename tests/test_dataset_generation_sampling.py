from tool_relevance_lab.dataset_generation.sampling import CandidateSampler, CandidateSamplingConfig
from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse


def make_universe(plugin_count: int = 20) -> ToolUniverse:
    return ToolUniverse(
        tool_universe_id="test_universe",
        tools=[
            ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
            ToolRecord(tool_id="agent.computer_use", kind="agent", source_text="computer"),
            *[
                ToolRecord(
                    tool_id=f"plugin.tool_{index:03d}",
                    kind="plugin",
                    source_text=f"plugin {index}",
                )
                for index in range(plugin_count)
            ],
        ],
    )


def test_candidate_sampler_includes_all_agents_and_plugin_subset() -> None:
    universe = make_universe(plugin_count=20)
    sampler = CandidateSampler(
        universe,
        CandidateSamplingConfig(
            plugin_sample_rate=0.10,
            target_force_rate=0.0,
            weight_decay_on_select=0.7,
        ),
    )

    record = sampler.sample(sample_id="sample_1", conversation_id="conv_1", seed=123)

    tool_ids = set(record.candidate_set.tool_ids)
    assert {"agent.browser_use", "agent.computer_use"}.issubset(tool_ids)
    assert len([tool_id for tool_id in tool_ids if tool_id.startswith("plugin.")]) == 2


def test_candidate_sampler_can_force_target_plugin() -> None:
    universe = make_universe(plugin_count=20)
    sampler = CandidateSampler(
        universe,
        CandidateSamplingConfig(
            plugin_sample_rate=0.10,
            target_force_rate=1.0,
            weight_decay_on_select=0.7,
        ),
    )

    record = sampler.sample(
        sample_id="sample_1",
        conversation_id="conv_1",
        seed=123,
        target_tool_ids=["plugin.tool_019"],
    )

    assert "plugin.tool_019" in record.candidate_set.tool_ids
    assert len([tool_id for tool_id in record.candidate_set.tool_ids if tool_id.startswith("plugin.")]) == 2


def test_candidate_sampler_reduces_selected_plugin_weights() -> None:
    universe = make_universe(plugin_count=20)
    sampler = CandidateSampler(
        universe,
        CandidateSamplingConfig(
            plugin_sample_rate=0.10,
            target_force_rate=1.0,
            weight_decay_on_select=0.5,
        ),
    )

    sampler.sample(
        sample_id="sample_1",
        conversation_id="conv_1",
        seed=123,
        target_tool_ids=["plugin.tool_019"],
    )

    assert sampler.exposure_counts["plugin.tool_019"] == 1
    assert sampler.weights["plugin.tool_019"] < 1.0
