from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr

V22B_SOURCE_DIR = Path(__file__).resolve().parents[3] / "modules" / "v22b_relevance_scorer" / "source"
if str(V22B_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(V22B_SOURCE_DIR))

from v22b_dataset_loader import PairExample, load_all_groups
from v22b_metrics import compute_metrics, write_json


ScenarioPredicate = Callable[[PairExample], bool]
ToolPredicate = Callable[[PairExample], bool]


SCENARIO_SLICES: dict[str, ScenarioPredicate] = {
    "no_tool_companion": lambda ex: ex.scenario_type
    in {
        "companion_chat_no_tool",
        "emotional_support_no_tool",
        "memory_recall_no_tool",
        "open_thread_followup_no_tool",
    },
    "boundary_refusal": lambda ex: ex.scenario_type == "boundary_or_refusal_no_tool",
    "agentic_task": lambda ex: ex.scenario_type == "agentic_task",
    "explicit_plugin_action": lambda ex: ex.scenario_type == "explicit_plugin_action",
    "weak_context": lambda ex: ex.scenario_type
    in {
        "assistant_suggested_tool_no_auth",
        "screen_context_weak_tool",
        "proactive_context_weak_tool",
        "passive_event",
    },
    "ambiguous": lambda ex: ex.scenario_type == "ambiguous_need_clarification",
}


TOOL_SLICES: dict[str, ToolPredicate] = {
    "agent": lambda ex: ex.tool_id.startswith("agent."),
    "plugin": lambda ex: ex.tool_id.startswith("plugin."),
    "external": lambda ex: _has_any(ex, ["send", "post", "reply", "upload", "api", "browser", "web", "发送", "发布", "上传"]),
    "monitoring": lambda ex: _has_any(ex, ["monitor", "listen", "watch", "camera", "detect", "监听", "监控", "观察"]),
    "proactive": lambda ex: _has_any(ex, ["proactive", "remind", "monitor", "watch", "listen", "auto", "提醒", "监听", "监控", "自动"]),
    "file": lambda ex: _has_any(ex, ["file", "folder", "pdf", "doc", "csv", "文件", "合同", "本地"]),
    "browser": lambda ex: _has_any(ex, ["browser", "web", "url", "网页", "页面", "链接", "打开"]),
    "search": lambda ex: _has_any(ex, ["search", "lookup", "find", "检索", "搜索", "查找"]),
    "message": lambda ex: _has_any(ex, ["sms", "discord", "wechat", "email", "reply", "message", "短信", "回复"]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="label=run_dir or run_dir")
    parser.add_argument("--run-root", type=Path, default=Path("tool_relevance/runtime/runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--high-label-threshold", type=float, default=1.0)
    parser.add_argument("--high-pred-threshold", type=float, default=1.0)
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    examples = _load_examples(config)
    runs = [_parse_run(value, args.run_root) for value in args.run]
    if not runs:
        raise ValueError("provide at least one --run label=run_dir")

    report = {
        "config": str(args.config),
        "runs": {},
    }
    for label, run_dir in runs:
        rows = _load_run_predictions(run_dir, examples)
        report["runs"][label] = analyze_rows(
            rows,
            topk=args.topk,
            high_label_threshold=args.high_label_threshold,
            high_pred_threshold=args.high_pred_threshold,
        )
    write_json(args.out, report)
    print(f"[done] wrote {args.out}")
    print_summary(report)


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    topk: int,
    high_label_threshold: float,
    high_pred_threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall": _group_metrics(
            rows,
            topk=topk,
            high_label_threshold=high_label_threshold,
            high_pred_threshold=high_pred_threshold,
        ),
        "scenario_slices": {},
        "scenario_exact": {},
        "relevance_slices": {},
        "tool_pair_slices": {},
        "tool_topk_exposure": {},
    }

    for name, predicate in SCENARIO_SLICES.items():
        subset = [row for row in rows if predicate(row["example"])]
        result["scenario_slices"][name] = _group_metrics(
            subset,
            topk=topk,
            high_label_threshold=high_label_threshold,
            high_pred_threshold=high_pred_threshold,
        )

    for scenario in sorted({row["example"].scenario_type for row in rows}):
        subset = [row for row in rows if row["example"].scenario_type == scenario]
        result["scenario_exact"][scenario] = _group_metrics(
            subset,
            topk=topk,
            high_label_threshold=high_label_threshold,
            high_pred_threshold=high_pred_threshold,
        )

    for relevance in sorted({row["example"].relevance_mode for row in rows}):
        subset = [row for row in rows if row["example"].relevance_mode == relevance]
        result["relevance_slices"][relevance] = _group_metrics(
            subset,
            topk=topk,
            high_label_threshold=high_label_threshold,
            high_pred_threshold=high_pred_threshold,
        )

    grouped = _group_rows(rows)
    for name, predicate in TOOL_SLICES.items():
        subset = [row for row in rows if predicate(row["example"])]
        result["tool_pair_slices"][name] = _pair_metrics(
            subset,
            high_label_threshold=high_label_threshold,
            high_pred_threshold=high_pred_threshold,
        )
        result["tool_topk_exposure"][name] = _tool_exposure(
            grouped,
            predicate,
            high_label_threshold=high_label_threshold,
        )
    return result


def _group_metrics(
    rows: list[dict[str, Any]],
    *,
    topk: int,
    high_label_threshold: float,
    high_pred_threshold: float,
) -> dict[str, float]:
    if not rows:
        return {"groups": 0.0, "pairs": 0.0}
    return compute_metrics(
        labels=np.asarray([row["label"] for row in rows], dtype="float32"),
        predictions=np.asarray([row["prediction"] for row in rows], dtype="float32"),
        conversation_ids=[row["conversation_id"] for row in rows],
        tool_ids=[row["tool_id"] for row in rows],
        high_label_threshold=high_label_threshold,
        high_pred_threshold=high_pred_threshold,
        topk=topk,
    )


def _pair_metrics(
    rows: list[dict[str, Any]],
    *,
    high_label_threshold: float,
    high_pred_threshold: float,
) -> dict[str, float]:
    if not rows:
        return {"pairs": 0.0}
    labels = np.asarray([row["label"] for row in rows], dtype="float32")
    predictions = np.asarray([row["prediction"] for row in rows], dtype="float32")
    corr = spearmanr(labels, predictions).statistic
    if np.isnan(corr):
        corr = 0.0
    high_pred = predictions >= high_pred_threshold
    high_label = labels >= high_label_threshold
    return {
        "pairs": float(len(rows)),
        "label_mean": float(labels.mean()),
        "prediction_mean": float(predictions.mean()),
        "mae": float(np.mean(np.abs(predictions - labels))),
        "spearman": float(corr),
        "high_conf_precision": float(np.logical_and(high_pred, high_label).sum() / max(1, high_pred.sum())),
        "high_value_recall": float(np.logical_and(high_pred, high_label).sum() / max(1, high_label.sum())),
        "bad_high_conf_rate": float(np.logical_and(high_pred, labels <= -high_label_threshold).sum() / max(1, high_pred.sum())),
    }


def _tool_exposure(
    grouped: dict[str, list[dict[str, Any]]],
    predicate: ToolPredicate,
    *,
    high_label_threshold: float,
) -> dict[str, float]:
    if not grouped:
        return {"groups": 0.0}
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    bad_top3_hits = 0
    bad_top5_hits = 0
    top3_count = 0
    top5_count = 0
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: row["prediction"], reverse=True)
        top1 = ordered[:1]
        top3 = ordered[:3]
        top5 = ordered[:5]
        top1_hits += int(any(predicate(row["example"]) for row in top1))
        top3_hits += int(any(predicate(row["example"]) for row in top3))
        top5_hits += int(any(predicate(row["example"]) for row in top5))
        bad_top3_hits += int(any(predicate(row["example"]) and row["label"] <= -high_label_threshold for row in top3))
        bad_top5_hits += int(any(predicate(row["example"]) and row["label"] <= -high_label_threshold for row in top5))
        top3_count += sum(1 for row in top3 if predicate(row["example"]))
        top5_count += sum(1 for row in top5 if predicate(row["example"]))
    group_count = len(grouped)
    return {
        "groups": float(group_count),
        "top1_exposure_rate": top1_hits / group_count,
        "top3_exposure_rate": top3_hits / group_count,
        "top5_exposure_rate": top5_hits / group_count,
        "bad_top3_exposure_rate": bad_top3_hits / group_count,
        "bad_top5_exposure_rate": bad_top5_hits / group_count,
        "avg_top3_count": top3_count / group_count,
        "avg_top5_count": top5_count / group_count,
    }


def _load_examples(config: dict[str, Any]) -> dict[tuple[str, str], PairExample]:
    examples = {}
    for group in load_all_groups(config):
        for example in group.examples:
            examples[(example.conversation_id, example.tool_id)] = example
    return examples


def _load_run_predictions(
    run_dir: Path,
    examples: dict[tuple[str, str], PairExample],
) -> list[dict[str, Any]]:
    prediction_paths = sorted(run_dir.glob("fold_*/best_val_predictions.jsonl"))
    if not prediction_paths:
        direct = run_dir / "best_val_predictions.jsonl"
        if direct.exists():
            prediction_paths = [direct]
    if not prediction_paths:
        raise FileNotFoundError(f"no best_val_predictions.jsonl found under {run_dir}")

    rows: list[dict[str, Any]] = []
    seen = set()
    for path in prediction_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["conversation_id"], row["tool_id"])
            if key in seen:
                raise ValueError(f"duplicate prediction row in {run_dir}: {key}")
            seen.add(key)
            row["example"] = examples[key]
            rows.append(row)
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)
    return grouped


def _parse_run(value: str, run_root: Path) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path = value
        label = Path(value).name
    run_dir = Path(path)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = run_root / path
    return label, run_dir


def _has_any(example: PairExample, patterns: list[str]) -> bool:
    text = f"{example.tool_id}\n{example.tool_text}".lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def print_summary(report: dict[str, Any]) -> None:
    keys = ["top1_match", "topk_recall", "ndcg_at_5", "top5_regret", "bad_top3_rate", "bad_top5_rate"]
    print("overall")
    print("run," + ",".join(keys))
    for label, payload in report["runs"].items():
        metrics = payload["overall"]
        print(label + "," + ",".join(_fmt(metrics.get(key)) for key in keys))
    print("\nkey scenario slices")
    for slice_name in SCENARIO_SLICES:
        print(f"[{slice_name}]")
        print("run,groups,top1,top3,ndcg5,regret5,bad3,bad5")
        for label, payload in report["runs"].items():
            metrics = payload["scenario_slices"][slice_name]
            print(
                ",".join(
                    [
                        label,
                        _fmt(metrics.get("groups")),
                        _fmt(metrics.get("top1_match")),
                        _fmt(metrics.get("topk_recall")),
                        _fmt(metrics.get("ndcg_at_5")),
                        _fmt(metrics.get("top5_regret")),
                        _fmt(metrics.get("bad_top3_rate")),
                        _fmt(metrics.get("bad_top5_rate")),
                    ]
                )
            )


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
