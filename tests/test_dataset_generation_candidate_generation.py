import pytest

from tool_relevance_lab.dataset_generation.candidate_generation import (
    CandidateGenerationConfig,
    generate_candidate_sets,
)
from tool_relevance_lab.dataset_generation.jsonl import append_jsonl, read_jsonl
from tool_relevance_lab.dataset_generation.schemas import (
    ConversationRecord,
    GenerationHint,
    Message,
    ToolRecord,
    ToolUniverse,
)


def make_universe(plugin_count: int = 20) -> ToolUniverse:
    return ToolUniverse(
        tool_universe_id="test_pool",
        tools=[
            ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
            ToolRecord(tool_id="agent.computer_use", kind="agent", source_text="computer"),
            *[
                ToolRecord(tool_id=f"plugin.tool_{index:03d}", kind="plugin", source_text="tool")
                for index in range(plugin_count)
            ],
        ],
    )


@pytest.mark.anyio
async def test_generate_candidate_sets_forces_target_and_resumes(tmp_path) -> None:
    conversations_path = tmp_path / "conversations.jsonl"
    output_path = tmp_path / "candidate_sets.jsonl"
    error_path = tmp_path / "errors.jsonl"
    append_jsonl(
        conversations_path,
        [
            ConversationRecord(
                conversation_id="conv_000001",
                messages=[Message(role="user", text="use a tool")],
                generation_hint=GenerationHint(target_tool_ids=["plugin.tool_019"]),
            )
        ],
    )

    config = CandidateGenerationConfig(
        plugin_sample_rate=0.10,
        target_force_rate=1.0,
        seed=123,
    )
    first = await generate_candidate_sets(
        universe=make_universe(),
        conversations_path=conversations_path,
        output_path=output_path,
        error_path=error_path,
        config=config,
    )
    second = await generate_candidate_sets(
        universe=make_universe(),
        conversations_path=conversations_path,
        output_path=output_path,
        error_path=error_path,
        config=config,
    )

    rows = list(read_jsonl(output_path))
    tool_ids = rows[0]["candidate_set"]["tool_ids"]
    assert rows[0]["sample_id"] == "sample_000001"
    assert "agent.browser_use" in tool_ids
    assert "agent.computer_use" in tool_ids
    assert "plugin.tool_019" in tool_ids
    assert len([tool_id for tool_id in tool_ids if tool_id.startswith("plugin.")]) == 2
    assert first.succeeded == 1
    assert second.skipped == 1
