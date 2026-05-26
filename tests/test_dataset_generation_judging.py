import pytest

from tool_relevance_lab.dataset_generation.jsonl import append_jsonl, read_jsonl
from tool_relevance_lab.dataset_generation.judging import JudgeGenerationConfig, judge_candidate_sets
from tool_relevance_lab.dataset_generation.schemas import (
    CandidateSet,
    CandidateSetRecord,
    ConversationRecord,
    GenerationHint,
    Message,
    Provenance,
    ToolRecord,
    ToolUniverse,
)


def make_universe() -> ToolUniverse:
    return ToolUniverse(
        tool_universe_id="test_pool",
        tools=[
            ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
            ToolRecord(tool_id="plugin.web_search", kind="plugin", source_text="web search"),
        ],
    )


@pytest.mark.anyio
async def test_judge_candidate_sets_dry_run_scores_all_candidates_and_resumes(tmp_path) -> None:
    conversations_path = tmp_path / "conversations.jsonl"
    candidate_sets_path = tmp_path / "candidate_sets.jsonl"
    output_path = tmp_path / "judgments.jsonl"
    error_path = tmp_path / "errors.jsonl"

    append_jsonl(
        conversations_path,
        [
            ConversationRecord(
                conversation_id="conv_000001",
                messages=[Message(role="user", text="search the web")],
                generation_hint=GenerationHint(target_tool_ids=["plugin.web_search"]),
                provenance=Provenance(extra={"tool_relevance_mode": "actionable_tool_relevance"}),
            )
        ],
    )
    append_jsonl(
        candidate_sets_path,
        [
            CandidateSetRecord(
                sample_id="sample_000001",
                conversation_id="conv_000001",
                candidate_set=CandidateSet(
                    sampling_seed=1,
                    plugin_sample_rate=1.0,
                    tool_ids=["agent.browser_use", "plugin.web_search"],
                ),
            )
        ],
    )

    config = JudgeGenerationConfig(judge_run_id="judge_test")
    first = await judge_candidate_sets(
        universe=make_universe(),
        client=None,
        conversations_path=conversations_path,
        candidate_sets_path=candidate_sets_path,
        output_path=output_path,
        error_path=error_path,
        config=config,
        dry_run=True,
    )
    second = await judge_candidate_sets(
        universe=make_universe(),
        client=None,
        conversations_path=conversations_path,
        candidate_sets_path=candidate_sets_path,
        output_path=output_path,
        error_path=error_path,
        config=config,
        dry_run=True,
    )

    rows = list(read_jsonl(output_path))
    assert rows[0]["sample_id"] == "sample_000001"
    assert rows[0]["candidate_tool_order"] == ["agent.browser_use", "plugin.web_search"]
    assert {score["tool_id"] for score in rows[0]["scores"]} == {
        "agent.browser_use",
        "plugin.web_search",
    }
    assert max(score["raw_score"] for score in rows[0]["scores"]) == 92.0
    assert first.succeeded == 1
    assert second.skipped == 1


@pytest.mark.anyio
async def test_judge_dry_run_keeps_no_tool_targets_low(tmp_path) -> None:
    conversations_path = tmp_path / "conversations.jsonl"
    candidate_sets_path = tmp_path / "candidate_sets.jsonl"
    output_path = tmp_path / "judgments.jsonl"
    error_path = tmp_path / "errors.jsonl"

    append_jsonl(
        conversations_path,
        [
            ConversationRecord(
                conversation_id="conv_000001",
                messages=[Message(role="user", text="我只是有点累，陪我说两句就好")],
                generation_hint=GenerationHint(target_tool_ids=["plugin.web_search"]),
                provenance=Provenance(extra={"tool_relevance_mode": "no_tool_or_low_relevance"}),
            )
        ],
    )
    append_jsonl(
        candidate_sets_path,
        [
            CandidateSetRecord(
                sample_id="sample_000001",
                conversation_id="conv_000001",
                candidate_set=CandidateSet(
                    sampling_seed=1,
                    plugin_sample_rate=1.0,
                    tool_ids=["agent.browser_use", "plugin.web_search"],
                ),
            )
        ],
    )

    await judge_candidate_sets(
        universe=make_universe(),
        client=None,
        conversations_path=conversations_path,
        candidate_sets_path=candidate_sets_path,
        output_path=output_path,
        error_path=error_path,
        config=JudgeGenerationConfig(judge_run_id="judge_test"),
        dry_run=True,
    )

    row = list(read_jsonl(output_path))[0]
    by_tool = {score["tool_id"]: score["raw_score"] for score in row["scores"]}
    assert by_tool["plugin.web_search"] == 24.0
