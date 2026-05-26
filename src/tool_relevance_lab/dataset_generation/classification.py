from __future__ import annotations

import re
from dataclasses import dataclass

from tool_relevance_lab.dataset_generation.schemas import (
    ToolCategory,
    ToolClassification,
    ToolRecord,
    ToolUniverse,
)


TAXONOMY_VERSION = "tool_taxonomy_v1"


@dataclass(frozen=True)
class CategoryRule:
    category: str
    tags: tuple[str, ...]
    patterns: tuple[str, ...]


DEFAULT_CATEGORY_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        category="web_retrieval",
        tags=("web", "search", "browser", "download"),
        patterns=("搜索", "search", "web", "网页", "浏览器", "browser", "download", "下载"),
    ),
    CategoryRule(
        category="file_document",
        tags=("file", "document", "pdf", "read_write"),
        patterns=("文件", "file", "document", "pdf", "docx", "excel", "csv", "读取", "写入"),
    ),
    CategoryRule(
        category="image_media",
        tags=("image", "media", "vision"),
        patterns=("图片", "image", "photo", "vision", "ocr", "截图", "视频", "video", "media"),
    ),
    CategoryRule(
        category="code_dev",
        tags=("code", "developer", "repository"),
        patterns=("代码", "code", "github", "git", "repo", "开发", "program", "debug", "测试"),
    ),
    CategoryRule(
        category="productivity",
        tags=("calendar", "email", "note", "task"),
        patterns=("日历", "calendar", "邮件", "email", "todo", "任务", "note", "笔记", "提醒"),
    ),
    CategoryRule(
        category="data_analysis",
        tags=("data", "analysis", "table"),
        patterns=("数据", "data", "分析", "analysis", "表格", "table", "统计", "chart", "图表"),
    ),
    CategoryRule(
        category="local_system",
        tags=("desktop", "os", "automation"),
        patterns=("电脑", "desktop", "computer", "操作系统", "应用", "窗口", "本地", "automation"),
    ),
    CategoryRule(
        category="communication",
        tags=("chat", "social", "message"),
        patterns=("消息", "message", "chat", "聊天", "社交", "social", "slack", "discord"),
    ),
)


MANUAL_TOOL_CATEGORY_OVERRIDES: dict[str, ToolCategory] = {
    "agent.qwenpaw": ToolCategory(
        tool_id="agent.qwenpaw",
        kind="agent",
        primary_category="remote_autonomous_agent",
        generation_tags=["agent", "remote", "autonomous", "long_running", "research", "web_workflow"],
        source="manual",
    ),
    "agent.openfang": ToolCategory(
        tool_id="agent.openfang",
        kind="agent",
        primary_category="tool_orchestration_agent",
        generation_tags=["agent", "wasm", "tool_orchestration", "code", "data_processing"],
        source="manual",
    ),
    "agent.browser_use": ToolCategory(
        tool_id="agent.browser_use",
        kind="agent",
        primary_category="web_retrieval",
        generation_tags=["agent", "browser", "web", "search", "download", "form_fill"],
        source="manual",
    ),
    "agent.computer_use": ToolCategory(
        tool_id="agent.computer_use",
        kind="agent",
        primary_category="local_system",
        generation_tags=["agent", "desktop", "os", "keyboard_mouse", "native_gui"],
        source="manual",
    ),
    "plugin.bilibili_danmaku": ToolCategory(
        tool_id="plugin.bilibili_danmaku",
        kind="plugin",
        primary_category="live_stream_monitoring",
        secondary_categories=["communication"],
        generation_tags=["bilibili", "danmaku", "live_stream", "passive", "message"],
        source="manual",
    ),
    "plugin.bilibili_dm": ToolCategory(
        tool_id="plugin.bilibili_dm",
        kind="plugin",
        primary_category="messaging",
        secondary_categories=["image_media"],
        generation_tags=["bilibili", "dm", "private_message", "auto_reply", "passive", "image"],
        source="manual",
    ),
    "plugin.galgame_plugin": ToolCategory(
        tool_id="plugin.galgame_plugin",
        kind="plugin",
        primary_category="game_assistant",
        secondary_categories=["image_media", "local_system"],
        generation_tags=["galgame", "visual_novel", "ocr", "game_state", "choice", "story"],
        source="manual",
    ),
    "plugin.game_agent_minecraft": ToolCategory(
        tool_id="plugin.game_agent_minecraft",
        kind="plugin",
        primary_category="game_assistant",
        generation_tags=["minecraft", "game_agent", "inventory", "game_state", "autonomous"],
        source="manual",
    ),
    "plugin.lifekit": ToolCategory(
        tool_id="plugin.lifekit",
        kind="plugin",
        primary_category="life_services",
        secondary_categories=["weather_location", "travel_local", "unit_currency"],
        generation_tags=[
            "weather",
            "forecast",
            "travel",
            "route",
            "nearby",
            "food",
            "recipe",
            "air_quality",
            "currency",
            "unit_convert",
            "countdown",
        ],
        source="manual",
    ),
    "plugin.mcp_adapter": ToolCategory(
        tool_id="plugin.mcp_adapter",
        kind="plugin",
        primary_category="tool_gateway",
        generation_tags=["mcp", "adapter", "gateway", "external_tools", "server"],
        source="manual",
    ),
    "plugin.memo_reminder": ToolCategory(
        tool_id="plugin.memo_reminder",
        kind="plugin",
        primary_category="reminder_scheduler",
        secondary_categories=["productivity"],
        generation_tags=["reminder", "alarm", "timer", "schedule", "memo", "recurring"],
        source="manual",
    ),
    "plugin.mijia": ToolCategory(
        tool_id="plugin.mijia",
        kind="plugin",
        primary_category="smart_home",
        generation_tags=["mijia", "xiaomi", "smart_home", "device", "light", "ac", "sensor", "scene"],
        source="manual",
    ),
    "plugin.proactive_controller": ToolCategory(
        tool_id="plugin.proactive_controller",
        kind="plugin",
        primary_category="system_control",
        secondary_categories=["communication"],
        generation_tags=["proactive_chat", "mode", "frequency", "controller", "settings"],
        source="manual",
    ),
    "plugin.qq_auto_reply": ToolCategory(
        tool_id="plugin.qq_auto_reply",
        kind="plugin",
        primary_category="messaging",
        generation_tags=["qq", "onebot", "auto_reply", "chat", "permission", "passive"],
        source="manual",
    ),
    "plugin.sts2_autoplay": ToolCategory(
        tool_id="plugin.sts2_autoplay",
        kind="plugin",
        primary_category="game_assistant",
        generation_tags=["slay_the_spire", "sts2", "autoplay", "game_state", "strategy", "commentary"],
        source="manual",
    ),
    "plugin.study_companion": ToolCategory(
        tool_id="plugin.study_companion",
        kind="plugin",
        primary_category="study_learning",
        secondary_categories=["image_media", "productivity"],
        generation_tags=["study", "ocr", "tutor", "question", "concept", "summary", "pomodoro"],
        source="manual",
    ),
    "plugin.web_search": ToolCategory(
        tool_id="plugin.web_search",
        kind="plugin",
        primary_category="web_retrieval",
        generation_tags=["web", "search", "baidu", "duckduckgo", "lookup"],
        source="manual",
    ),
}


def classify_tool_universe(
    universe: ToolUniverse,
    *,
    taxonomy_version: str = TAXONOMY_VERSION,
    rules: tuple[CategoryRule, ...] = DEFAULT_CATEGORY_RULES,
) -> ToolClassification:
    return ToolClassification(
        tool_universe_id=universe.tool_universe_id,
        taxonomy_version=taxonomy_version,
        categories=[classify_tool(tool, rules=rules) for tool in universe.tools],
    )


def classify_tool(
    tool: ToolRecord,
    *,
    rules: tuple[CategoryRule, ...] = DEFAULT_CATEGORY_RULES,
) -> ToolCategory:
    if tool.tool_id in MANUAL_TOOL_CATEGORY_OVERRIDES:
        override = MANUAL_TOOL_CATEGORY_OVERRIDES[tool.tool_id]
        if override.kind == tool.kind:
            return override

    if tool.kind == "agent":
        return _classify_agent(tool)

    text = _normalize_text(f"{tool.tool_id}\n{tool.source_text}")
    matched: list[CategoryRule] = []
    for rule in rules:
        if any(re.search(re.escape(pattern.lower()), text) for pattern in rule.patterns):
            matched.append(rule)

    if not matched:
        return ToolCategory(
            tool_id=tool.tool_id,
            kind=tool.kind,
            primary_category="other",
            generation_tags=[],
            source="heuristic",
        )

    primary = matched[0]
    tags = sorted({tag for rule in matched for tag in rule.tags})
    return ToolCategory(
        tool_id=tool.tool_id,
        kind=tool.kind,
        primary_category=primary.category,
        secondary_categories=[rule.category for rule in matched[1:]],
        generation_tags=tags,
        source="heuristic",
    )


def _classify_agent(tool: ToolRecord) -> ToolCategory:
    text = _normalize_text(f"{tool.tool_id}\n{tool.source_text}")
    if "browser" in text or "网页" in text or "浏览器" in text:
        return ToolCategory(
            tool_id=tool.tool_id,
            kind=tool.kind,
            primary_category="web_retrieval",
            secondary_categories=[],
            generation_tags=["agent", "browser", "web"],
            source="heuristic",
        )
    if "computer" in text or "desktop" in text or "电脑" in text or "操作系统" in text:
        return ToolCategory(
            tool_id=tool.tool_id,
            kind=tool.kind,
            primary_category="local_system",
            secondary_categories=[],
            generation_tags=["agent", "desktop", "os"],
            source="heuristic",
        )
    return ToolCategory(
        tool_id=tool.tool_id,
        kind=tool.kind,
        primary_category="agent_other",
        secondary_categories=[],
        generation_tags=["agent"],
        source="heuristic",
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def validate_classification_coverage(
    universe: ToolUniverse,
    classification: ToolClassification,
) -> None:
    universe_ids = {tool.tool_id for tool in universe.tools}
    classified_ids = {category.tool_id for category in classification.categories}
    missing = universe_ids - classified_ids
    extra = classified_ids - universe_ids
    if missing:
        raise ValueError(f"classification missing tool ids: {sorted(missing)}")
    if extra:
        raise ValueError(f"classification contains unknown tool ids: {sorted(extra)}")
