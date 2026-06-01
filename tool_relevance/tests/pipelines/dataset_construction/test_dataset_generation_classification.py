import pytest

from tool_relevance_lab.dataset_generation.classification import (
    classify_tool_universe,
    validate_classification_coverage,
)
from tool_relevance_lab.dataset_generation.schemas import ToolCategory, ToolRecord, ToolUniverse


def test_classify_tool_universe_covers_agents_and_plugins() -> None:
    universe = ToolUniverse(
        tool_universe_id="test_universe",
        tools=[
            ToolRecord(
                tool_id="agent.browser_use",
                kind="agent",
                source_text="本地浏览器自动化，适合打开 URL、填写网页表单、网页搜索。",
            ),
            ToolRecord(
                tool_id="agent.computer_use",
                kind="agent",
                source_text="本地电脑自动化，可操作桌面应用和操作系统。",
            ),
            ToolRecord(
                tool_id="plugin.web_search",
                kind="plugin",
                source_text="联网搜索，返回网页搜索结果。",
            ),
            ToolRecord(
                tool_id="plugin.pdf_reader",
                kind="plugin",
                source_text="读取 PDF 文件和文档内容。",
            ),
        ],
    )

    classification = classify_tool_universe(universe)
    validate_classification_coverage(universe, classification)
    by_id = classification.by_id()

    assert by_id["agent.browser_use"].primary_category == "web_retrieval"
    assert by_id["agent.computer_use"].primary_category == "local_system"
    assert by_id["plugin.web_search"].primary_category == "web_retrieval"
    assert by_id["plugin.pdf_reader"].primary_category == "file_document"


def test_validate_classification_coverage_rejects_missing_tool() -> None:
    universe = ToolUniverse(
        tool_universe_id="test_universe",
        tools=[
            ToolRecord(tool_id="agent.browser_use", kind="agent", source_text="browser"),
            ToolRecord(tool_id="plugin.web_search", kind="plugin", source_text="search"),
        ],
    )
    classification = classify_tool_universe(universe)
    classification.categories = [
        ToolCategory(
            tool_id="agent.browser_use",
            kind="agent",
            primary_category="web_retrieval",
            source="manual",
        )
    ]

    with pytest.raises(ValueError, match="missing"):
        validate_classification_coverage(universe, classification)
