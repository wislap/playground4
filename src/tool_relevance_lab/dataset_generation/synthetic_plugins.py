from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tool_relevance_lab.dataset_generation.llm import LLMClient, LLMRequest
from tool_relevance_lab.dataset_generation.neko_importer import build_plugin_source_text
from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse, ToolUniverseManifest
from tool_relevance_lab.dataset_generation.tool_store import write_tool_universe_dir


PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
FORBIDDEN_SYNTHETIC_FIELDS = {
    "runtime",
    "plugin_runtime",
    "status",
    "available",
    "source_path",
    "enabled",
    "auto_start",
}


class SyntheticPluginSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str = "plugin"
    description: str
    short_description: str
    keywords: list[str] = Field(default_factory=list)
    passive: bool = False

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if value.startswith("plugin."):
            value = value.removeprefix("plugin.")
        if not PLUGIN_ID_RE.match(value):
            raise ValueError("id must match ^[a-z][a-z0-9_]{2,63}$")
        return value

    @field_validator("name", "description", "short_description")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text fields must be non-empty")
        return value

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("keywords must be non-empty")
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="before")
    @classmethod
    def reject_runtime_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = sorted(FORBIDDEN_SYNTHETIC_FIELDS & set(data))
            if forbidden:
                raise ValueError(f"synthetic plugin must not contain runtime fields: {forbidden}")
        return data


class SyntheticPluginBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugins: list[SyntheticPluginSpec]


@dataclass(frozen=True)
class SyntheticPluginGenerationConfig:
    output_universe_id: str = "synthetic_plugins_v1"
    count: int = 40
    batch_size: int = 10
    model: str = "generator-model"
    temperature: float = 0.8
    prompt_version: str = "synthetic_plugin_generator_v1"
    creative_brief: str = ""


def synthetic_plugin_to_tool(spec: SyntheticPluginSpec) -> ToolRecord:
    source_text = build_plugin_source_text(
        plugin_id=spec.id,
        name=spec.name,
        plugin_type=spec.type,
        description=spec.description,
        short_description=spec.short_description,
        keywords=spec.keywords,
        passive=spec.passive,
    )
    return ToolRecord(
        tool_id=f"plugin.{spec.id}",
        kind="plugin",
        source_text=source_text,
        metadata=spec.model_dump(mode="json"),
    )


def validate_synthetic_tools(
    specs: list[SyntheticPluginSpec],
    *,
    existing_tool_ids: set[str],
) -> list[SyntheticPluginSpec]:
    seen: set[str] = set()
    accepted: list[SyntheticPluginSpec] = []
    for spec in specs:
        tool_id = f"plugin.{spec.id}"
        if tool_id in existing_tool_ids or tool_id in seen:
            continue
        seen.add(tool_id)
        accepted.append(spec)
    return accepted


def build_synthetic_plugin_prompt(
    *,
    real_universe: ToolUniverse,
    taxonomy_text: str,
    count: int,
    batch_index: int,
    existing_ids: list[str],
    creative_brief: str = "",
) -> list[dict[str, str]]:
    examples = _render_plugin_examples(real_universe, limit=12)
    system = (
        "You generate realistic N.E.K.O-style plugin metadata for a synthetic tool universe. "
        "Synthetic plugins are not installed and have no runtime state. "
        "Return strict JSON only."
    )
    user = (
        f"Generate {count} synthetic plugin records for batch {batch_index}.\n\n"
        "Use these real plugins as style examples. Do not copy them:\n"
        f"{examples}\n\n"
        "Current taxonomy and coverage hints:\n"
        f"{taxonomy_text}\n\n"
        f"{_render_creative_brief(creative_brief)}"
        "Existing plugin ids that must not be reused:\n"
        f"{', '.join(existing_ids)}\n\n"
        "Rules:\n"
        "- Each plugin must be realistic, concrete, and naturally invokable by a user request.\n"
        "- Use Chinese names/descriptions when natural; short_description can be English or bilingual.\n"
        "- Include useful Chinese and English keywords.\n"
        "- passive=true only for background listeners, auto-reply bridges, or event monitors.\n"
        "- id is the bare plugin id only, for example daily_briefing; never prefix it with plugin.\n"
        "- Do not include runtime, plugin_runtime, status, available, enabled, auto_start, source_path, "
        "entries, schemas, examples, risk, side effects, or confirmation policies.\n"
        "- Output exactly this JSON shape: {\"plugins\": [{\"id\": \"...\", \"name\": \"...\", "
        "\"type\": \"plugin\", \"description\": \"...\", \"short_description\": \"...\", "
        "\"keywords\": [\"...\"], \"passive\": false}]}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def generate_synthetic_plugin_universe(
    *,
    client: LLMClient,
    real_universe: ToolUniverse,
    existing_universes: list[ToolUniverse] | None = None,
    taxonomy_text: str,
    output_dir: Path,
    config: SyntheticPluginGenerationConfig,
) -> ToolUniverse:
    existing_tool_ids = {tool.tool_id for tool in real_universe.tools}
    for universe in existing_universes or []:
        existing_tool_ids.update(tool.tool_id for tool in universe.tools)
    accepted: list[SyntheticPluginSpec] = []
    batch_index = 0
    max_batches = max(3, (config.count // max(config.batch_size, 1) + 1) * 3)

    while len(accepted) < config.count and batch_index < max_batches:
        batch_index += 1
        remaining = config.count - len(accepted)
        batch_count = min(config.batch_size, remaining)
        messages = build_synthetic_plugin_prompt(
            real_universe=real_universe,
            taxonomy_text=taxonomy_text,
            count=batch_count,
            batch_index=batch_index,
            existing_ids=sorted(existing_tool_ids | {f"plugin.{item.id}" for item in accepted}),
            creative_brief=config.creative_brief,
        )
        request = LLMRequest(
            request_id=f"synthetic_plugins.{config.output_universe_id}.batch_{batch_index}",
            idempotency_key=f"{config.output_universe_id}:{config.prompt_version}:batch:{batch_index}",
            messages=messages,
            temperature=config.temperature,
            response_format={"type": "json_object"},
        )
        response = await client.complete(request)
        parsed = SyntheticPluginBatch.model_validate(_json_object_from_text(response.text))
        new_specs = validate_synthetic_tools(
            parsed.plugins,
            existing_tool_ids=existing_tool_ids | {f"plugin.{item.id}" for item in accepted},
        )
        accepted.extend(new_specs[:remaining])

    if len(accepted) < config.count:
        raise RuntimeError(f"only generated {len(accepted)} unique synthetic plugins")

    universe = ToolUniverse(
        tool_universe_id=config.output_universe_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        tools=[synthetic_plugin_to_tool(spec) for spec in accepted],
    )
    manifest = ToolUniverseManifest(
        tool_universe_id=universe.tool_universe_id,
        created_at=universe.created_at,
        kind="synthetic",
        tool_count=len(universe.tools),
        source={
            "type": "llm_generated",
            "model": config.model,
            "prompt_version": config.prompt_version,
            "real_universe_id": real_universe.tool_universe_id,
        },
    )
    write_tool_universe_dir(output_dir, universe, manifest)
    return universe


def _render_plugin_examples(real_universe: ToolUniverse, *, limit: int) -> str:
    plugins = [tool for tool in real_universe.tools if tool.kind == "plugin"]
    lines: list[str] = []
    for tool in plugins[:limit]:
        meta = tool.metadata
        lines.append(
            json.dumps(
                {
                    "id": meta.get("id"),
                    "name": meta.get("name"),
                    "description": meta.get("description"),
                    "short_description": meta.get("short_description"),
                    "keywords": meta.get("keywords", []),
                    "passive": meta.get("passive", False),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _render_creative_brief(creative_brief: str) -> str:
    creative_brief = creative_brief.strip()
    if not creative_brief:
        return ""
    return (
        "Additional generation brief. Follow this to diversify the synthetic universe:\n"
        f"{creative_brief}\n\n"
    )


def _json_object_from_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data


def summarize_taxonomy_for_prompt(taxonomy_path: Path) -> str:
    raw = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    categories = raw.get("categories", [])
    counts = Counter(item.get("primary_category", "other") for item in categories if isinstance(item, dict))
    return "\n".join(f"- {category}: {count}" for category, count in sorted(counts.items()))
