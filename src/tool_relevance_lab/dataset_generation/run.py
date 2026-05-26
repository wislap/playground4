from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from tool_relevance_lab.dataset_generation.neko_importer import (
    build_neko_tool_universe,
    build_neko_tool_universe_manifest,
    write_tool_universe,
)
from tool_relevance_lab.dataset_generation.classification import (
    classify_tool_universe,
    validate_classification_coverage,
)
from tool_relevance_lab.dataset_generation.calibration import calibrate_confidences
from tool_relevance_lab.dataset_generation.jsonl import read_model_jsonl, write_jsonl
from tool_relevance_lab.dataset_generation.llm import OpenAICompatibleConfig, OpenAICompatibleLLMClient
from tool_relevance_lab.dataset_generation.quality import summarize_quality
from tool_relevance_lab.dataset_generation.schemas import (
    CandidateSetRecord,
    JudgmentRecord,
)
from tool_relevance_lab.dataset_generation.similarity import find_similar_tools, summarize_similarity
from tool_relevance_lab.dataset_generation.synthetic_plugins import (
    SyntheticPluginGenerationConfig,
    generate_synthetic_plugin_universe,
    summarize_taxonomy_for_prompt,
)
from tool_relevance_lab.dataset_generation.tool_store import (
    load_tool_universes,
    read_tool_universe,
    write_tool_universe_dir,
)
from tool_relevance_lab.dataset_generation.universe_ops import (
    merge_tool_universes,
    validate_synthetic_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset generation pipeline utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--judgments", "--judgements", dest="judgments", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--clip-percentile", type=float, default=0.001)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--tool-universe", type=Path, action="append", required=True)
    report_parser.add_argument("--tool-universe-id")
    report_parser.add_argument("--candidate-sets", type=Path, required=True)
    report_parser.add_argument("--judgments", "--judgements", dest="judgments", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)

    classify_parser = subparsers.add_parser("classify-tools")
    classify_parser.add_argument("--tool-universe", type=Path, action="append", required=True)
    classify_parser.add_argument("--tool-universe-id")
    classify_parser.add_argument("--output", type=Path, required=True)

    inspect_parser = subparsers.add_parser("inspect-tool-universes")
    inspect_parser.add_argument("--tool-universe", type=Path, action="append", required=True)
    inspect_parser.add_argument("--tool-universe-id")
    inspect_parser.add_argument("--output", type=Path)

    similarity_parser = subparsers.add_parser("report-tool-similarity")
    similarity_parser.add_argument("--tool-universe", type=Path, action="append", required=True)
    similarity_parser.add_argument("--tool-universe-id")
    similarity_parser.add_argument("--output", type=Path, required=True)
    similarity_parser.add_argument("--min-score", type=float, default=0.28)
    similarity_parser.add_argument("--min-text-cosine", type=float, default=0.42)
    similarity_parser.add_argument("--min-keyword-jaccard", type=float, default=0.30)
    similarity_parser.add_argument("--min-id-family-similarity", type=float, default=0.50)
    similarity_parser.add_argument("--include-agents", action="store_true")

    import_neko_parser = subparsers.add_parser("import-neko-tools")
    import_neko_parser.add_argument("--neko-root", type=Path, required=True)
    import_neko_parser.add_argument("--output", type=Path, required=True)
    import_neko_parser.add_argument("--tool-universe-id", default="tool_universe_v1")

    synthetic_parser = subparsers.add_parser("generate-synthetic-plugins")
    synthetic_parser.add_argument("--real-tool-universe", type=Path, required=True)
    synthetic_parser.add_argument("--existing-tool-universe", type=Path, action="append", default=[])
    synthetic_parser.add_argument("--taxonomy", type=Path, required=True)
    synthetic_parser.add_argument("--output", type=Path, required=True)
    synthetic_parser.add_argument("--tool-universe-id", default="synthetic_plugins_v1")
    synthetic_parser.add_argument("--count", type=int, default=40)
    synthetic_parser.add_argument("--batch-size", type=int, default=10)
    synthetic_parser.add_argument("--model", default=os.getenv("TOOL_GEN_MODEL", "generator-model"))
    synthetic_parser.add_argument("--base-url", default=os.getenv("TOOL_GEN_BASE_URL", ""))
    synthetic_parser.add_argument("--api-key", default=os.getenv("TOOL_GEN_API_KEY", ""))
    synthetic_parser.add_argument("--temperature", type=float, default=0.8)
    synthetic_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    synthetic_parser.add_argument("--retries", type=int, default=3)
    synthetic_parser.add_argument("--creative-brief", default="")

    validate_synthetic_parser = subparsers.add_parser("validate-synthetic-plugins")
    validate_synthetic_parser.add_argument("--synthetic-tool-universe", type=Path, required=True)
    validate_synthetic_parser.add_argument("--real-tool-universe", type=Path)
    validate_synthetic_parser.add_argument("--output", type=Path)

    merge_parser = subparsers.add_parser("merge-tool-universes")
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--tool-universe-id", required=True)
    merge_parser.add_argument("inputs", type=Path, nargs="+")

    args = parser.parse_args()
    if args.command == "calibrate":
        judgments = read_model_jsonl(args.judgments, JudgmentRecord)
        scores = calibrate_confidences(judgments, clip_percentile=args.clip_percentile)
        write_jsonl(args.output, [score.__dict__ for score in scores])
        return

    if args.command == "report":
        universe = load_tool_universes(args.tool_universe, tool_universe_id=args.tool_universe_id)
        candidate_sets = read_model_jsonl(args.candidate_sets, CandidateSetRecord)
        judgments = read_model_jsonl(args.judgments, JudgmentRecord)
        summary = summarize_quality(
            universe=universe,
            candidate_sets=candidate_sets,
            judgments=judgments,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary.__dict__, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    if args.command == "classify-tools":
        universe = load_tool_universes(args.tool_universe, tool_universe_id=args.tool_universe_id)
        classification = classify_tool_universe(universe)
        validate_classification_coverage(universe, classification)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                classification.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return

    if args.command == "inspect-tool-universes":
        universe = load_tool_universes(args.tool_universe, tool_universe_id=args.tool_universe_id)
        payload = {
            "tool_universe_id": universe.tool_universe_id,
            "tool_count": len(universe.tools),
            "agent_count": len(universe.agents),
            "plugin_count": len(universe.plugins),
            "tool_ids": [tool.tool_id for tool in universe.tools],
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "report-tool-similarity":
        universe = load_tool_universes(args.tool_universe, tool_universe_id=args.tool_universe_id)
        pairs = find_similar_tools(
            universe,
            min_score=args.min_score,
            min_text_cosine=args.min_text_cosine,
            min_keyword_jaccard=args.min_keyword_jaccard,
            min_id_family_similarity=args.min_id_family_similarity,
            plugins_only=not args.include_agents,
        )
        payload = summarize_similarity(universe, pairs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    if args.command == "import-neko-tools":
        universe = build_neko_tool_universe(
            neko_root=args.neko_root,
            tool_universe_id=args.tool_universe_id,
        )
        if args.output.suffix:
            write_tool_universe(args.output, universe)
        else:
            manifest = build_neko_tool_universe_manifest(universe=universe, neko_root=args.neko_root)
            write_tool_universe_dir(args.output, universe, manifest)
        print(f"wrote {len(universe.tools)} tools to {args.output}")
        return

    if args.command == "generate-synthetic-plugins":
        if not args.api_key or not args.base_url:
            raise SystemExit("TOOL_GEN_API_KEY/TOOL_GEN_BASE_URL or --api-key/--base-url are required")
        real_universe = read_tool_universe(args.real_tool_universe)
        existing_universes = [read_tool_universe(path) for path in args.existing_tool_universe]
        taxonomy_text = summarize_taxonomy_for_prompt(args.taxonomy)
        client = OpenAICompatibleLLMClient(
            OpenAICompatibleConfig(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.retries,
            )
        )
        config = SyntheticPluginGenerationConfig(
            output_universe_id=args.tool_universe_id,
            count=args.count,
            batch_size=args.batch_size,
            model=args.model,
            temperature=args.temperature,
            creative_brief=args.creative_brief,
        )
        universe = asyncio.run(
            generate_synthetic_plugin_universe(
                client=client,
                real_universe=real_universe,
                existing_universes=existing_universes,
                taxonomy_text=taxonomy_text,
                output_dir=args.output,
                config=config,
            )
        )
        print(f"wrote {len(universe.tools)} synthetic plugins to {args.output}")
        return

    if args.command == "validate-synthetic-plugins":
        synthetic_universe = read_tool_universe(args.synthetic_tool_universe)
        real_universe = read_tool_universe(args.real_tool_universe) if args.real_tool_universe else None
        report = validate_synthetic_universe(
            synthetic_universe,
            real_universe=real_universe,
        )
        payload = report.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if not report.ok:
            raise SystemExit(1)
        return

    if args.command == "merge-tool-universes":
        universes = [read_tool_universe(path) for path in args.inputs]
        merged = merge_tool_universes(
            output_dir=args.output,
            output_universe_id=args.tool_universe_id,
            universes=universes,
        )
        print(f"wrote {len(merged.tools)} tools to {args.output}")
        return


if __name__ == "__main__":
    main()
