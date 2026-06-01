from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse, ToolUniverseManifest
from tool_relevance_lab.dataset_generation.tool_store import write_tool_universe_dir


AGENT_CHANNELS: tuple[dict[str, Any], ...] = (
    {
        "id": "qwenpaw",
        "method": "openclaw",
        "enable_flag": "openclaw_enabled",
        "capability_key": "openclaw",
        "priority": 0,
        "description": (
            "- **qwenpaw**: 远程 Agent 系统 + 云端虚拟机。"
            "最适合需要完全自主的复杂、长时间任务（如多步研究、复杂网页工作流）。"
            "最慢最贵，但最强大。"
        ),
    },
    {
        "id": "openfang",
        "method": "openfang",
        "enable_flag": "openfang_enabled",
        "capability_key": "openfang",
        "priority": 1,
        "description": (
            "- **openfang**: 本地 WASM 沙箱多 Agent 系统。"
            "适合需要工具编排的复合任务（数据处理、代码执行、多步思考、多维检索）。"
            "比浏览器慢但功能强大。不适合需要屏幕/GUI 交互的任务。"
        ),
    },
    {
        "id": "browser_use",
        "method": "browser_use",
        "enable_flag": "browser_use_enabled",
        "capability_key": "browser_use",
        "priority": 2,
        "description": (
            "- **browser_use**: 本地浏览器自动化。"
            "快速且经济，适合简单网页交互：打开 URL、填写网页表单、网页搜索、从网络下载。"
            "仅限本地浏览器任务 - 无法与操作系统应用交互。"
            "如果任务能在网页内完成，应优先选择它，而不是 `computer_use`。"
        ),
    },
    {
        "id": "computer_use",
        "method": "computer_use",
        "enable_flag": "computer_use_enabled",
        "capability_key": "computer_use",
        "priority": 3,
        "description": (
            "- **computer_use**: 直接控制本地键盘和鼠标。"
            "唯一可以与本地操作系统交互的渠道（打开桌面应用、点击原生 UI 元素、控制鼠标键盘）。"
            "较慢、较贵，且会占用用户的鼠标键盘。"
            "在任务明确需要本地操作系统 GUI 交互时使用。"
            "如果网页内就能完成，不要优先选择它。"
        ),
    },
)


def build_neko_tool_universe(
    *,
    neko_root: Path,
    tool_universe_id: str = "tool_universe_v1",
) -> ToolUniverse:
    plugin_root = neko_root / "plugin" / "plugins"
    tools = [build_agent_tool(channel) for channel in AGENT_CHANNELS]
    tools.extend(build_plugin_tool(path) for path in sorted(plugin_root.glob("*/plugin.toml")))
    return ToolUniverse(
        tool_universe_id=tool_universe_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        tools=tools,
    )


def build_neko_tool_universe_manifest(
    *,
    universe: ToolUniverse,
    neko_root: Path,
) -> ToolUniverseManifest:
    return ToolUniverseManifest(
        tool_universe_id=universe.tool_universe_id,
        created_at=universe.created_at,
        kind="real",
        tool_count=len(universe.tools),
        source={
            "type": "neko_repo",
            "path": str(neko_root),
            "importer": "neko_importer_v1",
        },
    )


def build_agent_tool(channel: dict[str, Any]) -> ToolRecord:
    source_text = "\n".join(
        [
            "kind: agent",
            f"id: {channel['id']}",
            f"method: {channel['method']}",
            f"enable_flag: {channel['enable_flag']}",
            f"capability_key: {channel['capability_key']}",
            f"priority: {channel['priority']}",
            f"description: {channel['description']}",
        ]
    )
    return ToolRecord(
        tool_id=f"agent.{channel['id']}",
        kind="agent",
        source_text=source_text,
        metadata={
            "id": channel["id"],
            "method": channel["method"],
            "enable_flag": channel["enable_flag"],
            "capability_key": channel["capability_key"],
            "priority": channel["priority"],
            "description": channel["description"],
        },
    )


def build_plugin_tool(plugin_toml_path: Path) -> ToolRecord:
    raw = tomllib.loads(plugin_toml_path.read_text(encoding="utf-8"))
    plugin = raw.get("plugin", raw)
    translations = _load_plugin_translations(plugin_toml_path.parent, plugin)

    plugin_id = _clean_str(plugin.get("id")) or plugin_toml_path.parent.name
    name = _resolve_i18n_value(_clean_str(plugin.get("name")), translations) or plugin_id
    plugin_type = _clean_str(plugin.get("type")) or "plugin"
    description = _resolve_i18n_value(_clean_str(plugin.get("description")), translations)
    short_description = _resolve_i18n_value(_clean_str(plugin.get("short_description")), translations)
    keywords = _clean_keywords(plugin.get("keywords"))
    passive = bool(plugin.get("passive", False))

    source_text = build_plugin_source_text(
        plugin_id=plugin_id,
        name=name,
        plugin_type=plugin_type,
        description=description,
        short_description=short_description,
        keywords=keywords,
        passive=passive,
    )
    return ToolRecord(
        tool_id=f"plugin.{plugin_id}",
        kind="plugin",
        source_text=source_text,
        metadata={
            "id": plugin_id,
            "name": name,
            "type": plugin_type,
            "description": description,
            "short_description": short_description,
            "keywords": keywords,
            "passive": passive,
        },
    )


def build_plugin_source_text(
    *,
    plugin_id: str,
    name: str,
    plugin_type: str,
    description: str,
    short_description: str,
    keywords: list[str],
    passive: bool,
) -> str:
    return "\n".join(
        [
            "kind: plugin",
            f"id: {plugin_id}",
            f"name: {name}",
            f"type: {plugin_type}",
            f"description: {description}",
            f"short_description: {short_description}",
            f"keywords: {', '.join(keywords)}",
            f"passive: {str(passive).lower()}",
        ]
    )


def write_tool_universe(path: Path, universe: ToolUniverse) -> None:
    if path.suffix == "":
        manifest = ToolUniverseManifest(
            tool_universe_id=universe.tool_universe_id,
            created_at=universe.created_at,
            kind="real",
            tool_count=len(universe.tools),
            source={"importer": "neko_importer_v1"},
        )
        write_tool_universe_dir(path, universe, manifest)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(universe.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _clean_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _load_plugin_translations(plugin_dir: Path, plugin: dict[str, Any]) -> dict[str, str]:
    i18n = plugin.get("i18n")
    if not isinstance(i18n, dict):
        return {}
    locale = _clean_str(i18n.get("default_locale")) or "zh-CN"
    locales_dir = _clean_str(i18n.get("locales_dir")) or "i18n"
    locale_path = plugin_dir / locales_dir / f"{locale}.json"
    if not locale_path.exists():
        return {}
    try:
        data = json.loads(locale_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if isinstance(value, str)}


def _resolve_i18n_value(value: str, translations: dict[str, str]) -> str:
    if value in translations:
        return translations[value].strip()
    return value
