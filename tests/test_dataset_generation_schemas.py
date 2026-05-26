import pytest

from tool_relevance_lab.dataset_generation.schemas import JudgmentRecord, RawToolScore, ToolRecord, ToolUniverse


def test_tool_universe_rejects_duplicate_tool_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        ToolUniverse(
            tool_universe_id="bad",
            tools=[
                ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
                ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
            ],
        )


def test_judgment_requires_scores_to_cover_candidate_order() -> None:
    with pytest.raises(ValueError, match="cover"):
        JudgmentRecord(
            sample_id="sample_1",
            conversation_id="conv_1",
            judge_run_id="run_1",
            candidate_tool_order=["a", "b"],
            scores=[RawToolScore(tool_id="a", raw_score=10)],
        )
