from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def compute_metrics(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    conversation_ids: list[str],
    high_label_threshold: float = 0.8,
    high_pred_threshold: float = 0.8,
    topk: int = 3,
) -> dict[str, float]:
    error = predictions - labels
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, conversation_id in enumerate(conversation_ids):
        grouped[conversation_id].append(index)

    top1_hits = 0
    topk_hits = 0
    no_tool_groups = 0
    no_tool_fp = 0
    high_pred = 0
    high_pred_correct = 0

    for indices in grouped.values():
        label_values = labels[indices]
        pred_values = predictions[indices]
        best_label = int(np.argmax(label_values))
        pred_order = list(np.argsort(-pred_values))
        if pred_order[0] == best_label:
            top1_hits += 1
        if best_label in pred_order[:topk]:
            topk_hits += 1
        if float(label_values.max()) < high_label_threshold:
            no_tool_groups += 1
            if float(pred_values.max()) >= high_pred_threshold:
                no_tool_fp += 1
        for label, pred in zip(label_values, pred_values, strict=True):
            if float(pred) >= high_pred_threshold:
                high_pred += 1
                if float(label) >= high_label_threshold:
                    high_pred_correct += 1

    corr = spearmanr(labels, predictions).statistic
    if np.isnan(corr):
        corr = 0.0
    group_count = max(1, len(grouped))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error * error)),
        "spearman": float(corr),
        "top1_match": top1_hits / group_count,
        "topk_recall": topk_hits / group_count,
        "no_tool_fp_rate": no_tool_fp / max(1, no_tool_groups),
        "high_conf_precision": high_pred_correct / max(1, high_pred),
        "groups": float(len(grouped)),
        "pairs": float(len(labels)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
