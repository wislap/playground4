from pathlib import Path

import pytest

from tool_relevance_lab.dataset_generation.conversation_generation import (
    ConversationGenerationConfig,
    _sanitize_conversation_payload,
    build_conversation_tasks,
    generate_conversations,
)
from tool_relevance_lab.dataset_generation.jsonl import read_jsonl
from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse


def make_universe() -> ToolUniverse:
    return ToolUniverse(
        tool_universe_id="test_pool",
        tools=[
            ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
            ToolRecord(
                tool_id="plugin.web_search",
                kind="plugin",
                source_text="web search",
                metadata={"name": "网页搜索"},
            ),
            ToolRecord(
                tool_id="plugin.pdf_reader",
                kind="plugin",
                source_text="pdf reader",
                metadata={"name": "PDF 阅读器"},
            ),
        ],
    )


def test_build_conversation_tasks_is_stable() -> None:
    universe = make_universe()
    config = ConversationGenerationConfig(count=3, seed=123)

    first = build_conversation_tasks(universe=universe, config=config)
    second = build_conversation_tasks(universe=universe, config=config)

    assert first == second
    assert [task.conversation_id for task in first] == [
        "conv_000001",
        "conv_000002",
        "conv_000003",
    ]
    assert all(task.scenario_type for task in first)


def test_sanitize_conversation_payload_normalizes_attachment_objects() -> None:
    payload = {
        "conversation_id": "conv_000001",
        "messages": [
            {
                "role": "user",
                "text": "帮我看这两个文件",
                "attachments": [
                    {"name": "eval_A.csv", "type": "text/csv"},
                    {"filename": "screen.png", "mime_type": "image/png"},
                ],
            }
        ],
    }

    sanitized = _sanitize_conversation_payload(payload)

    assert sanitized["messages"][0]["attachments"] == [
        "eval_A.csv (text/csv)",
        "screen.png (image/png)",
    ]


@pytest.mark.anyio
async def test_generate_conversations_dry_run_resumes(tmp_path: Path) -> None:
    universe = make_universe()
    output_path = tmp_path / "conversations.jsonl"
    error_path = tmp_path / "errors.jsonl"

    config = ConversationGenerationConfig(count=4, seed=123, concurrency=2)
    first = await generate_conversations(
        universe=universe,
        client=None,
        output_path=output_path,
        error_path=error_path,
        config=config,
        dry_run=True,
    )
    second = await generate_conversations(
        universe=universe,
        client=None,
        output_path=output_path,
        error_path=error_path,
        config=config,
        dry_run=True,
    )

    rows = list(read_jsonl(output_path))
    assert len(rows) == 4
    assert rows[0]["conversation_id"] == "conv_000001"
    assert rows[0]["generation_hint"]["target_tool_ids"]
    assert "scenario_type" in rows[0]["provenance"]["extra"]
    assert "tool_relevance_mode" in rows[0]["provenance"]["extra"]
    assert first.succeeded == 4
    assert second.skipped == 4
    assert second.submitted == 0
