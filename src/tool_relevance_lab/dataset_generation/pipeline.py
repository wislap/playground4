from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tool_relevance_lab.dataset_generation.calibration import calibrate_confidences
from tool_relevance_lab.dataset_generation.candidate_generation import (
    CandidateGenerationConfig,
    generate_candidate_sets,
)
from tool_relevance_lab.dataset_generation.conversation_generation import (
    ConversationGenerationConfig,
    generate_conversations,
)
from tool_relevance_lab.dataset_generation.jsonl import read_model_jsonl, write_jsonl
from tool_relevance_lab.dataset_generation.judging import (
    JudgeGenerationConfig,
    judge_candidate_sets,
)
from tool_relevance_lab.dataset_generation.llm import (
    OpenAICompatibleConfig,
    OpenAICompatibleLLMClient,
)
from tool_relevance_lab.dataset_generation.multiaxis import (
    calibrate_multiaxis_confidences,
    judge_candidate_sets_multiaxis,
    multiaxis_rows_to_jsonl,
)
from tool_relevance_lab.dataset_generation.quality import summarize_quality
from tool_relevance_lab.dataset_generation.resumable import ResumableRunSummary, write_run_summary
from tool_relevance_lab.dataset_generation.schemas import CandidateSetRecord, JudgmentRecord, MultiAxisJudgmentRecord
from tool_relevance_lab.dataset_generation.tool_store import load_tool_universes


@dataclass(frozen=True)
class APIClientConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    retries: int = 3


@dataclass(frozen=True)
class PipelineConfig:
    run_id: str = "neko_v3_500"
    output_root: Path = Path("data")
    tool_universe_paths: tuple[Path, ...] = (
        Path("data/tool_universes/neko_real_v1"),
        Path("data/tool_universes/synthetic_plugins_v1"),
        Path("data/tool_universes/synthetic_plugins_v2"),
        Path("data/tool_universes/synthetic_plugins_v3"),
        Path("data/tool_universes/synthetic_plugins_v4"),
    )
    tool_universe_id: str = "runtime_tool_pool_v4"
    count: int = 500
    seed: int = 20260526
    targets_per_conversation: int = 1
    conversation_concurrency: int = 6
    candidate_concurrency: int = 1
    judge_concurrency: int = 8
    max_attempts: int = 3
    plugin_sample_rate: float = 0.10
    target_force_rate: float = 1.0
    weight_decay_on_select: float = 0.70
    judge_run_id: str = "judge_v3"
    judge_mode: str = "single"
    clip_percentile: float = 0.001
    generator: APIClientConfig = field(default_factory=APIClientConfig)
    judge: APIClientConfig = field(default_factory=APIClientConfig)


@dataclass(frozen=True)
class PipelinePaths:
    run_dir: Path
    conversations: Path
    conversation_errors: Path
    conversation_summary: Path
    candidate_sets: Path
    candidate_errors: Path
    candidate_summary: Path
    judgments: Path
    judgment_errors: Path
    judgment_summary: Path
    calibrated: Path
    quality_report: Path
    manifest: Path


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    paths: PipelinePaths
    conversation_summary: ResumableRunSummary
    candidate_summary: ResumableRunSummary
    judgment_summary: ResumableRunSummary
    calibrated_rows: int
    quality_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "paths": {key: str(value) for key, value in self.paths.__dict__.items()},
            "conversation_summary": self.conversation_summary.to_dict(),
            "candidate_summary": self.candidate_summary.to_dict(),
            "judgment_summary": self.judgment_summary.to_dict(),
            "calibrated_rows": self.calibrated_rows,
            "quality_report": self.quality_report,
        }


def load_pipeline_config(path: Path) -> PipelineConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.get("run", {})
    tool_universe = raw.get("tool_universe", {})
    generation = raw.get("generation", {})
    sampling = raw.get("candidate_sampling", {})
    calibration = raw.get("calibration", {})
    llm = raw.get("llm", {})

    output_root = Path(str(run.get("output_root", "data")))
    tool_paths = tuple(Path(str(item)) for item in tool_universe.get("paths", []))

    return PipelineConfig(
        run_id=str(run.get("run_id", "neko_v3_500")),
        output_root=output_root,
        tool_universe_paths=tool_paths or PipelineConfig().tool_universe_paths,
        tool_universe_id=str(tool_universe.get("id", "runtime_tool_pool_v4")),
        count=int(generation.get("count", 500)),
        seed=int(generation.get("seed", run.get("seed", 20260526))),
        targets_per_conversation=int(generation.get("targets_per_conversation", 1)),
        conversation_concurrency=int(generation.get("concurrency", 6)),
        candidate_concurrency=int(sampling.get("concurrency", 1)),
        judge_concurrency=int(llm.get("judge", {}).get("concurrency", 8)),
        max_attempts=int(run.get("max_attempts", 3)),
        plugin_sample_rate=float(sampling.get("plugin_sample_rate", 0.10)),
        target_force_rate=float(sampling.get("target_force_rate", 1.0)),
        weight_decay_on_select=float(sampling.get("weight_decay_on_select", 0.70)),
        judge_run_id=str(llm.get("judge", {}).get("judge_run_id", "judge_v3")),
        judge_mode=str(llm.get("judge", {}).get("mode", "single")),
        clip_percentile=float(calibration.get("clip_percentile", 0.001)),
        generator=_api_config(
            llm.get("generator", {}),
            env_prefix="CONV_GEN",
            default_model="generator-model",
            default_temperature=0.9,
        ),
        judge=_api_config(
            llm.get("judge", {}),
            env_prefix="JUDGE",
            default_model="judge-model",
            default_temperature=0.2,
        ),
    )


def pipeline_paths(config: PipelineConfig) -> PipelinePaths:
    run_dir = config.output_root / "runs" / config.run_id
    return PipelinePaths(
        run_dir=run_dir,
        conversations=run_dir / "conversations.jsonl",
        conversation_errors=run_dir / "errors" / "conversations.errors.jsonl",
        conversation_summary=run_dir / "summaries" / "conversations.summary.json",
        candidate_sets=run_dir / "candidate_sets.jsonl",
        candidate_errors=run_dir / "errors" / "candidate_sets.errors.jsonl",
        candidate_summary=run_dir / "summaries" / "candidate_sets.summary.json",
        judgments=run_dir / "judgments.jsonl",
        judgment_errors=run_dir / "errors" / "judgments.errors.jsonl",
        judgment_summary=run_dir / "summaries" / "judgments.summary.json",
        calibrated=run_dir / "calibrated.jsonl",
        quality_report=run_dir / "quality_report.json",
        manifest=run_dir / "manifest.json",
    )


async def run_dataset_pipeline(
    *,
    config: PipelineConfig,
    dry_run: bool = False,
) -> PipelineResult:
    if config.count < 1:
        raise ValueError("pipeline count must be >= 1")

    paths = pipeline_paths(config)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(paths.manifest, config=config, dry_run=dry_run, status="running")

    universe = load_tool_universes(
        list(config.tool_universe_paths),
        tool_universe_id=config.tool_universe_id,
    )
    generator_client = None if dry_run else _build_client(config.generator, "generator")
    judge_client = None if dry_run else _build_client(config.judge, "judge")

    print(f"[pipeline] run_id={config.run_id} tools={len(universe.tools)} count={config.count}")
    print(f"[pipeline] stage 1/5 generate conversations -> {paths.conversations}")
    conversation_summary = await generate_conversations(
        universe=universe,
        client=generator_client,
        output_path=paths.conversations,
        error_path=paths.conversation_errors,
        config=ConversationGenerationConfig(
            count=config.count,
            targets_per_conversation=config.targets_per_conversation,
            seed=config.seed,
            model=config.generator.model or "generator-model",
            temperature=config.generator.temperature,
            concurrency=config.conversation_concurrency,
            max_attempts=config.max_attempts,
        ),
        dry_run=dry_run,
    )
    write_run_summary(paths.conversation_summary, conversation_summary)

    print(f"[pipeline] stage 2/5 sample candidates -> {paths.candidate_sets}")
    candidate_summary = await generate_candidate_sets(
        universe=universe,
        conversations_path=paths.conversations,
        output_path=paths.candidate_sets,
        error_path=paths.candidate_errors,
        config=CandidateGenerationConfig(
            plugin_sample_rate=config.plugin_sample_rate,
            target_force_rate=config.target_force_rate,
            weight_decay_on_select=config.weight_decay_on_select,
            seed=config.seed,
            concurrency=config.candidate_concurrency,
        ),
    )
    write_run_summary(paths.candidate_summary, candidate_summary)

    print(f"[pipeline] stage 3/5 judge candidates ({config.judge_mode}) -> {paths.judgments}")
    judge_config = JudgeGenerationConfig(
        model=config.judge.model or "judge-model",
        temperature=config.judge.temperature,
        concurrency=config.judge_concurrency,
        max_attempts=config.max_attempts,
        judge_run_id=config.judge_run_id,
        prompt_version=(
            "neko_tool_multiaxis_judge_v1"
            if config.judge_mode == "multiaxis"
            else "neko_tool_relevance_judge_v3"
        ),
    )
    if config.judge_mode == "single":
        judgment_summary = await judge_candidate_sets(
            universe=universe,
            client=judge_client,
            conversations_path=paths.conversations,
            candidate_sets_path=paths.candidate_sets,
            output_path=paths.judgments,
            error_path=paths.judgment_errors,
            config=judge_config,
            dry_run=dry_run,
        )
    elif config.judge_mode == "multiaxis":
        judgment_summary = await judge_candidate_sets_multiaxis(
            universe=universe,
            client=judge_client,
            conversations_path=paths.conversations,
            candidate_sets_path=paths.candidate_sets,
            output_path=paths.judgments,
            error_path=paths.judgment_errors,
            config=judge_config,
            dry_run=dry_run,
        )
    else:
        raise ValueError(f"unknown judge_mode: {config.judge_mode}")
    write_run_summary(paths.judgment_summary, judgment_summary)

    print(f"[pipeline] stage 4/5 calibrate -> {paths.calibrated}")
    if config.judge_mode == "single":
        judgments = read_model_jsonl(paths.judgments, JudgmentRecord)
        calibrated = calibrate_confidences(judgments, clip_percentile=config.clip_percentile)
        write_jsonl(paths.calibrated, [row.__dict__ for row in calibrated])
        calibrated_rows = len(calibrated)
    else:
        multiaxis_judgments = read_model_jsonl(paths.judgments, MultiAxisJudgmentRecord)
        calibrated_multiaxis = calibrate_multiaxis_confidences(
            multiaxis_judgments,
            clip_percentile=config.clip_percentile,
        )
        write_jsonl(paths.calibrated, multiaxis_rows_to_jsonl(calibrated_multiaxis))
        judgments = _multiaxis_to_final_judgments(multiaxis_judgments)
        calibrated_rows = len(calibrated_multiaxis)

    print(f"[pipeline] stage 5/5 quality report -> {paths.quality_report}")
    candidate_sets = read_model_jsonl(paths.candidate_sets, CandidateSetRecord)
    quality = summarize_quality(
        universe=universe,
        candidate_sets=candidate_sets,
        judgments=judgments,
    )
    quality_payload = quality.__dict__
    paths.quality_report.parent.mkdir(parents=True, exist_ok=True)
    paths.quality_report.write_text(
        json.dumps(quality_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = PipelineResult(
        run_id=config.run_id,
        paths=paths,
        conversation_summary=conversation_summary,
        candidate_summary=candidate_summary,
        judgment_summary=judgment_summary,
        calibrated_rows=calibrated_rows,
        quality_report=quality_payload,
    )
    _write_manifest(paths.manifest, config=config, dry_run=dry_run, status="completed", result=result)
    print(f"[pipeline] completed run_id={config.run_id}")
    return result


def _api_config(
    raw: dict[str, Any],
    *,
    env_prefix: str,
    default_model: str,
    default_temperature: float,
) -> APIClientConfig:
    return APIClientConfig(
        base_url=str(raw.get("base_url") or os.getenv(f"{env_prefix}_BASE_URL", "")),
        api_key=str(raw.get("api_key") or os.getenv(f"{env_prefix}_API_KEY", "")),
        model=str(raw.get("model") or os.getenv(f"{env_prefix}_MODEL", default_model)),
        temperature=float(raw.get("temperature", default_temperature)),
        timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
        retries=int(raw.get("retries", 3)),
    )


def _build_client(config: APIClientConfig, label: str) -> OpenAICompatibleLLMClient:
    if _is_placeholder(config.api_key) or _is_placeholder(config.base_url):
        raise RuntimeError(
            f"{label} LLM is not configured. Set TOML llm.{label}.api_key/base_url "
            f"or matching environment variables."
        )
    return OpenAICompatibleLLMClient(
        OpenAICompatibleConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.retries,
        )
    )


def _multiaxis_to_final_judgments(judgments: list[MultiAxisJudgmentRecord]) -> list[JudgmentRecord]:
    from tool_relevance_lab.dataset_generation.schemas import RawToolScore

    converted = []
    for judgment in judgments:
        converted.append(
            JudgmentRecord(
                sample_id=judgment.sample_id,
                conversation_id=judgment.conversation_id,
                judge_run_id=judgment.judge_run_id,
                candidate_tool_order=judgment.candidate_tool_order,
                scores=[
                    RawToolScore(
                        tool_id=tool_score.tool_id,
                        raw_score=next(score.raw_score for score in tool_score.scores if score.axis == "final_preference"),
                    )
                    for tool_score in judgment.scores
                ],
                provenance=judgment.provenance,
            )
        )
    return converted


def _is_placeholder(value: str) -> bool:
    stripped = (value or "").strip()
    return stripped in {"", "sk-...", "https://your-relay.example/v1"}


def _safe_config_dump(config: PipelineConfig) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "output_root": str(config.output_root),
        "tool_universe_paths": [str(path) for path in config.tool_universe_paths],
        "tool_universe_id": config.tool_universe_id,
        "count": config.count,
        "seed": config.seed,
        "targets_per_conversation": config.targets_per_conversation,
        "conversation_concurrency": config.conversation_concurrency,
        "candidate_concurrency": config.candidate_concurrency,
        "judge_concurrency": config.judge_concurrency,
        "max_attempts": config.max_attempts,
        "plugin_sample_rate": config.plugin_sample_rate,
        "target_force_rate": config.target_force_rate,
        "weight_decay_on_select": config.weight_decay_on_select,
        "judge_run_id": config.judge_run_id,
        "judge_mode": config.judge_mode,
        "clip_percentile": config.clip_percentile,
        "generator": _safe_api_dump(config.generator),
        "judge": _safe_api_dump(config.judge),
    }


def _safe_api_dump(config: APIClientConfig) -> dict[str, Any]:
    return {
        "base_url": config.base_url,
        "api_key_set": bool(config.api_key),
        "model": config.model,
        "temperature": config.temperature,
        "timeout_seconds": config.timeout_seconds,
        "retries": config.retries,
    }


def _write_manifest(
    path: Path,
    *,
    config: PipelineConfig,
    dry_run: bool,
    status: str,
    result: PipelineResult | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "dry_run": dry_run,
        "config": _safe_config_dump(config),
    }
    if result is not None:
        payload["result"] = result.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
