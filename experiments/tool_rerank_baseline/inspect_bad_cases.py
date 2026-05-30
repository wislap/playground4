from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from data import PairExample, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    bundle = load_dataset(config)
    examples = {(_key(example)): example for example in bundle.val_examples}
    predictions_path = args.predictions or _default_predictions_path(args.run_dir)
    predictions = [json.loads(line) for line in predictions_path.open(encoding="utf-8")]
    for row in predictions:
        row["example"] = examples[(row["conversation_id"], row["tool_id"])]

    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["conversation_id"]].append(row)

    top1_misses = []
    for rows in grouped.values():
        gold = max(rows, key=lambda row: row["label"])
        pred = max(rows, key=lambda row: row["prediction"])
        if gold["tool_id"] != pred["tool_id"]:
            top1_misses.append(
                {
                    "kind": "top1_miss",
                    "conversation_id": gold["conversation_id"],
                    "severity": gold["label"] - pred["label"],
                    "gold": gold,
                    "pred": pred,
                    "rows": rows,
                }
            )
    top1_misses.sort(key=lambda item: item["severity"], reverse=True)

    false_positives = sorted(
        [
            {
                "kind": "false_positive",
                "severity": row["prediction"] - row["label"],
                "row": row,
                "rows": grouped[row["conversation_id"]],
            }
            for row in predictions
            if row["label"] < -1.0 and row["prediction"] > 0.0
        ],
        key=lambda item: item["severity"],
        reverse=True,
    )
    false_negatives = sorted(
        [
            {
                "kind": "false_negative",
                "severity": row["label"] - row["prediction"],
                "row": row,
                "rows": grouped[row["conversation_id"]],
            }
            for row in predictions
            if row["label"] > 1.0 and row["prediction"] < 0.0
        ],
        key=lambda item: item["severity"],
        reverse=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render_report(
            run_dir=args.run_dir,
            predictions_path=predictions_path,
            top1_misses=top1_misses[: args.limit],
            false_positives=false_positives[: args.limit],
            false_negatives=false_negatives[: args.limit],
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {args.out}")


def render_report(
    *,
    run_dir: Path,
    predictions_path: Path,
    top1_misses: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
) -> str:
    parts = [
        "# Bad Case Inspection",
        "",
        f"Run: `{run_dir}`",
        f"Predictions: `{predictions_path}`",
        "",
        "## Top1 Ranking Misses",
        "",
    ]
    for index, item in enumerate(top1_misses, start=1):
        parts.append(render_top1_case(index, item))
    parts.extend(["", "## False Positives", ""])
    for index, item in enumerate(false_positives, start=1):
        parts.append(render_single_case(index, item["row"], item["rows"], title="False Positive"))
    parts.extend(["", "## False Negatives", ""])
    for index, item in enumerate(false_negatives, start=1):
        parts.append(render_single_case(index, item["row"], item["rows"], title="False Negative"))
    return "\n".join(parts)


def render_top1_case(index: int, item: dict[str, Any]) -> str:
    gold = item["gold"]
    pred = item["pred"]
    example: PairExample = gold["example"]
    return "\n".join(
        [
            f"### {index}. `{gold['conversation_id']}`",
            "",
            f"- Predicted top1: `{pred['tool_id']}` label={pred['label']:.3f} pred={pred['prediction']:.3f}",
            f"- Gold top1: `{gold['tool_id']}` label={gold['label']:.3f} pred={gold['prediction']:.3f}",
            f"- Scenario: `{example.scenario_type}` / `{example.relevance_mode}`",
            "",
            "#### Conversation",
            "",
            fenced(example.conversation_text),
            "",
            "#### Candidate Ranking",
            "",
            ranking_table(item["rows"]),
            "",
            "#### Predicted Tool",
            "",
            fenced(pred["example"].tool_text),
            "",
            "#### Gold Tool",
            "",
            fenced(gold["example"].tool_text),
            "",
        ]
    )


def render_single_case(index: int, row: dict[str, Any], rows: list[dict[str, Any]], *, title: str) -> str:
    example: PairExample = row["example"]
    return "\n".join(
        [
            f"### {index}. {title}: `{row['conversation_id']}` / `{row['tool_id']}`",
            "",
            f"- label={row['label']:.3f} pred={row['prediction']:.3f} raw={row['raw_score']:.1f}",
            f"- Scenario: `{example.scenario_type}` / `{example.relevance_mode}`",
            "",
            "#### Conversation",
            "",
            fenced(example.conversation_text),
            "",
            "#### Candidate Ranking",
            "",
            ranking_table(rows),
            "",
            "#### Tool",
            "",
            fenced(example.tool_text),
            "",
        ]
    )


def ranking_table(rows: list[dict[str, Any]], *, limit: int = 8) -> str:
    ordered = sorted(rows, key=lambda row: row["prediction"], reverse=True)[:limit]
    lines = ["| rank | tool | label | pred | raw |", "| ---: | --- | ---: | ---: | ---: |"]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | `{row['tool_id']}` | {row['label']:.3f} | "
            f"{row['prediction']:.3f} | {row['raw_score']:.1f} |"
        )
    return "\n".join(lines)


def fenced(text: str) -> str:
    return f"```text\n{text.strip()}\n```"


def _key(example: PairExample) -> tuple[str, str]:
    return example.conversation_id, example.tool_id


def _default_predictions_path(run_dir: Path) -> Path:
    best_path = run_dir / "best_val_predictions.jsonl"
    if best_path.exists():
        return best_path
    return run_dir / "val_predictions.jsonl"


if __name__ == "__main__":
    main()
