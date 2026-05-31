from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

try:
    from data import ASK_CONFIRM_ID, NO_TOOL_ID
except ImportError:  # pragma: no cover - package/script import compatibility
    ASK_CONFIRM_ID = "__ASK_CONFIRM__"
    NO_TOOL_ID = "__NO_TOOL__"


NON_CALL_TOOL_IDS = {NO_TOOL_ID, ASK_CONFIRM_ID}


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
    bad_exposed_top1_groups = 0
    bad_exposed_top3_groups = 0
    bad_exposed_top5_groups = 0
    active_bad_top1_groups = 0
    active_bad_top3_groups = 0
    active_bad_top5_groups = 0
    no_tool_suppressed_bad_top3_groups = 0
    no_tool_suppressed_bad_top5_groups = 0
    agent_bad_top3_groups = 0
    top3_label_means = []
    top5_label_means = []
    top3_label_maxes = []
    top5_label_maxes = []
    high_value_total = 0
    high_value_hit_at_3 = 0
    high_value_hit_at_5 = 0
    activation_groups = 0
    activation_hits = 0
    active_groups = 0
    active_tool_top1_hits = 0
    active_tool_topk_hits = 0
    no_tool_groups_with_candidate = 0
    no_tool_top1_hits = 0
    predicted_no_tool_groups = 0
    predicted_no_tool_correct = 0
    action_groups = 0
    action_hits = 0
    ask_confirm_groups = 0
    ask_confirm_top1_hits = 0
    predicted_ask_confirm_groups = 0
    predicted_ask_confirm_correct = 0

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
        local_tool_ids = [tool_ids[indices[index]] for index in range(len(indices))] if tool_ids is not None else []
        if tool_ids is not None and NO_TOOL_ID in local_tool_ids:
            no_tool_local_index = local_tool_ids.index(NO_TOOL_ID)
            gold_no_tool = best_label == no_tool_local_index
            pred_no_tool = pred_order[0] == no_tool_local_index
            activation_groups += 1
            activation_hits += int(gold_no_tool == pred_no_tool)
            predicted_no_tool_groups += int(pred_no_tool)
            predicted_no_tool_correct += int(pred_no_tool and gold_no_tool)
            if gold_no_tool:
                no_tool_groups_with_candidate += 1
                no_tool_top1_hits += int(pred_no_tool)
            else:
                active_groups += 1
                if pred_order[0] != no_tool_local_index and pred_order[0] == best_label:
                    active_tool_top1_hits += 1
                active_tool_topk_hits += int(
                    any(index != no_tool_local_index and index == best_label for index in pred_order[:topk])
                )
        if tool_ids is not None and local_tool_ids and any(tool_id in NON_CALL_TOOL_IDS for tool_id in local_tool_ids):
            gold_action = _activation_action(local_tool_ids[best_label])
            pred_action = _activation_action(local_tool_ids[pred_order[0]])
            action_groups += 1
            action_hits += int(gold_action == pred_action)
            if ASK_CONFIRM_ID in local_tool_ids:
                ask_confirm_local_index = local_tool_ids.index(ASK_CONFIRM_ID)
                gold_ask_confirm = best_label == ask_confirm_local_index
                pred_ask_confirm = pred_order[0] == ask_confirm_local_index
                predicted_ask_confirm_groups += int(pred_ask_confirm)
                predicted_ask_confirm_correct += int(pred_ask_confirm and gold_ask_confirm)
                if gold_ask_confirm:
                    ask_confirm_groups += 1
                    ask_confirm_top1_hits += int(pred_ask_confirm)
        top1 = pred_order[:1]
        top3 = pred_order[: min(3, len(pred_order))]
        top5 = pred_order[: min(5, len(pred_order))]
        exposed_top1 = top1
        exposed_top3 = top3
        exposed_top5 = top5
        if (
            tool_ids is not None
            and local_tool_ids
            and local_tool_ids[pred_order[0]] in NON_CALL_TOOL_IDS
        ):
            exposed_top1 = []
            exposed_top3 = []
            exposed_top5 = []
            no_tool_suppressed_bad_top3_groups += int(np.any(label_values[top3] <= -high_label_threshold))
            no_tool_suppressed_bad_top5_groups += int(np.any(label_values[top5] <= -high_label_threshold))
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
        if exposed_top1 and np.any(label_values[exposed_top1] <= -high_label_threshold):
            bad_exposed_top1_groups += 1
        if exposed_top3 and np.any(label_values[exposed_top3] <= -high_label_threshold):
            bad_exposed_top3_groups += 1
        if exposed_top5 and np.any(label_values[exposed_top5] <= -high_label_threshold):
            bad_exposed_top5_groups += 1
        if exposed_top1:
            active_bad_top1_groups += int(np.any(label_values[top1] <= -high_label_threshold))
            active_bad_top3_groups += int(np.any(label_values[top3] <= -high_label_threshold))
            active_bad_top5_groups += int(np.any(label_values[top5] <= -high_label_threshold))
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
        "bad_exposed_top1_rate": bad_exposed_top1_groups / group_count,
        "bad_exposed_top3_rate": bad_exposed_top3_groups / group_count,
        "bad_exposed_top5_rate": bad_exposed_top5_groups / group_count,
        "active_bad_top1_rate": active_bad_top1_groups / max(1, active_groups),
        "active_bad_top3_rate": active_bad_top3_groups / max(1, active_groups),
        "active_bad_top5_rate": active_bad_top5_groups / max(1, active_groups),
        "no_tool_suppressed_bad_top3_rate": no_tool_suppressed_bad_top3_groups / max(1, predicted_no_tool_groups),
        "no_tool_suppressed_bad_top5_rate": no_tool_suppressed_bad_top5_groups / max(1, predicted_no_tool_groups),
        "agent_bad_top3_rate": agent_bad_top3_groups / group_count,
        "top3_label_mean": float(np.mean(top3_label_means)),
        "top5_label_mean": float(np.mean(top5_label_means)),
        "top3_label_max": float(np.mean(top3_label_maxes)),
        "top5_label_max": float(np.mean(top5_label_maxes)),
        "high_value_recall_at_3": high_value_hit_at_3 / max(1, high_value_total),
        "high_value_recall_at_5": high_value_hit_at_5 / max(1, high_value_total),
        "no_tool_fp_rate": no_tool_fp / max(1, no_tool_groups),
        "high_conf_precision": high_pred_correct / max(1, high_pred),
        "activation_accuracy": activation_hits / max(1, activation_groups),
        "no_tool_recall": no_tool_top1_hits / max(1, no_tool_groups_with_candidate),
        "no_tool_precision": predicted_no_tool_correct / max(1, predicted_no_tool_groups),
        "active_tool_top1": active_tool_top1_hits / max(1, active_groups),
        "active_tool_topk": active_tool_topk_hits / max(1, active_groups),
        "activation_action_accuracy": action_hits / max(1, action_groups),
        "ask_confirm_recall": ask_confirm_top1_hits / max(1, ask_confirm_groups),
        "ask_confirm_precision": predicted_ask_confirm_correct / max(1, predicted_ask_confirm_groups),
        "ask_confirm_candidate_groups": float(ask_confirm_groups),
        "activation_groups": float(activation_groups),
        "no_tool_candidate_groups": float(no_tool_groups_with_candidate),
        "active_candidate_groups": float(active_groups),
        "groups": float(len(grouped)),
        "pairs": float(len(labels)),
    }


def _activation_action(tool_id: str) -> str:
    if tool_id == NO_TOOL_ID:
        return "no_tool"
    if tool_id == ASK_CONFIRM_ID:
        return "ask_confirm"
    return "call_tool"


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
