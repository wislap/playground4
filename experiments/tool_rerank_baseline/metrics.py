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
    tool_ids: list[str] | None = None,
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
    regrets_at_1 = []
    regrets_at_3 = []
    regrets_at_5 = []
    ndcg_at_3 = []
    ndcg_at_5 = []
    bad_top3_groups = 0
    bad_top5_groups = 0
    agent_bad_top3_groups = 0
    top3_label_means = []
    top5_label_means = []
    top3_label_maxes = []
    top5_label_maxes = []
    high_value_total = 0
    high_value_hit_at_3 = 0
    high_value_hit_at_5 = 0

    for indices in grouped.values():
        label_values = labels[indices]
        pred_values = predictions[indices]
        best_label = int(np.argmax(label_values))
        pred_order = list(np.argsort(-pred_values))
        best_label_value = float(label_values[best_label])
        if pred_order[0] == best_label:
            top1_hits += 1
        if best_label in pred_order[:topk]:
            topk_hits += 1
        top1 = pred_order[:1]
        top3 = pred_order[: min(3, len(pred_order))]
        top5 = pred_order[: min(5, len(pred_order))]
        regrets_at_1.append(best_label_value - float(label_values[top1].max()))
        regrets_at_3.append(best_label_value - float(label_values[top3].max()))
        regrets_at_5.append(best_label_value - float(label_values[top5].max()))
        ndcg_at_3.append(_ndcg_at_k(label_values, pred_order, k=3))
        ndcg_at_5.append(_ndcg_at_k(label_values, pred_order, k=5))
        top3_label_means.append(float(label_values[top3].mean()))
        top5_label_means.append(float(label_values[top5].mean()))
        top3_label_maxes.append(float(label_values[top3].max()))
        top5_label_maxes.append(float(label_values[top5].max()))
        if np.any(label_values[top3] <= -high_label_threshold):
            bad_top3_groups += 1
        if np.any(label_values[top5] <= -high_label_threshold):
            bad_top5_groups += 1
        if tool_ids is not None:
            for index in top3:
                global_index = indices[index]
                if label_values[index] <= -high_label_threshold and tool_ids[global_index].startswith("agent."):
                    agent_bad_top3_groups += 1
                    break
        high_value_indices = {int(index) for index in np.where(label_values >= high_label_threshold)[0]}
        high_value_total += len(high_value_indices)
        high_value_hit_at_3 += len(high_value_indices & set(top3))
        high_value_hit_at_5 += len(high_value_indices & set(top5))
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
        "top1_regret": float(np.mean(regrets_at_1)),
        "top3_regret": float(np.mean(regrets_at_3)),
        "top5_regret": float(np.mean(regrets_at_5)),
        "ndcg_at_3": float(np.mean(ndcg_at_3)),
        "ndcg_at_5": float(np.mean(ndcg_at_5)),
        "bad_top3_rate": bad_top3_groups / group_count,
        "bad_top5_rate": bad_top5_groups / group_count,
        "agent_bad_top3_rate": agent_bad_top3_groups / group_count,
        "top3_label_mean": float(np.mean(top3_label_means)),
        "top5_label_mean": float(np.mean(top5_label_means)),
        "top3_label_max": float(np.mean(top3_label_maxes)),
        "top5_label_max": float(np.mean(top5_label_maxes)),
        "high_value_recall_at_3": high_value_hit_at_3 / max(1, high_value_total),
        "high_value_recall_at_5": high_value_hit_at_5 / max(1, high_value_total),
        "no_tool_fp_rate": no_tool_fp / max(1, no_tool_groups),
        "high_conf_precision": high_pred_correct / max(1, high_pred),
        "groups": float(len(grouped)),
        "pairs": float(len(labels)),
    }


def _ndcg_at_k(label_values: np.ndarray, pred_order: list[int], *, k: int) -> float:
    top = pred_order[: min(k, len(pred_order))]
    ideal = list(np.argsort(-label_values))[: min(k, len(label_values))]
    gains = _positive_gains(label_values)
    dcg = _dcg(gains[top])
    ideal_dcg = _dcg(gains[ideal])
    if ideal_dcg <= 0:
        return 1.0 if dcg <= 0 else 0.0
    return float(dcg / ideal_dcg)


def _positive_gains(label_values: np.ndarray) -> np.ndarray:
    shifted = label_values - float(label_values.min())
    return np.maximum(shifted, 0.0)


def _dcg(gains: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains * discounts))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
