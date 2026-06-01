from pathlib import Path

import pytest

from tool_relevance_lab.dataset_generation.jsonl import read_jsonl
from tool_relevance_lab.dataset_generation.pipeline import (
    PipelineConfig,
    run_dataset_pipeline,
)
from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse, ToolUniverseManifest
from tool_relevance_lab.dataset_generation.tool_store import write_tool_universe_dir


def write_test_universe(path: Path) -> None:
    universe = ToolUniverse(
        tool_universe_id="test_pool",
        tools=[
            ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser agent"),
            ToolRecord(tool_id="plugin.web_search", kind="plugin", source_text="web search plugin"),
            ToolRecord(tool_id="plugin.calendar", kind="plugin", source_text="calendar plugin"),
        ],
    )
    write_tool_universe_dir(
        path,
        universe,
        ToolUniverseManifest(
            tool_universe_id="test_pool",
            kind="test",
            tool_count=len(universe.tools),
        ),
    )


@pytest.mark.anyio
async def test_run_dataset_pipeline_dry_run_resumes(tmp_path: Path) -> None:
    universe_path = tmp_path / "tool_universe"
    write_test_universe(universe_path)

    config = PipelineConfig(
        run_id="pipeline_test",
        output_root=tmp_path / "data",
        tool_universe_paths=(universe_path,),
        tool_universe_id="test_pool",
        count=3,
        conversation_concurrency=2,
        judge_concurrency=2,
    )

    first = await run_dataset_pipeline(config=config, dry_run=True)
    second = await run_dataset_pipeline(config=config, dry_run=True)

    assert first.conversation_summary.succeeded == 3
    assert second.conversation_summary.skipped == 3
    assert first.candidate_summary.succeeded == 3
    assert second.candidate_summary.skipped == 3
    assert first.judgment_summary.succeeded == 3
    assert second.judgment_summary.skipped == 3
    assert first.paths.manifest.exists()
    assert first.paths.quality_report.exists()
    assert len(list(read_jsonl(first.paths.conversations))) == 3
    assert len(list(read_jsonl(first.paths.candidate_sets))) == 3
    assert len(list(read_jsonl(first.paths.judgments))) == 3
    assert first.calibrated_rows > 0
