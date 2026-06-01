from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from v22b_dataset_loader import PairExample
from v22b_metrics import append_jsonl, compute_metrics, write_json
from v22b_model import (
    EncodedFieldSequencePairs,
    EncodedPairs,
    EncodedSequencePairs,
    FieldInteractionRegressor,
    FactorizedLateInteractionRegressor,
    LexicalFeatureBuilder,
    LateInteractionRegressor,
    PairMLPRegressor,
    build_pair_features,
)
from v22b_policy_features import PolicyFeatureBuilder

def encode_examples(examples: list[PairExample], encoder: Any, config: dict[str, Any]) -> EncodedPairs:
    batch_size = int(config["encoder"].get("batch_size", 64))
    conv_vectors = encoder.encode([example.conversation_text for example in examples], batch_size=batch_size)
    tool_vectors = encoder.encode([example.tool_text for example in examples], batch_size=batch_size)
    return EncodedPairs(
        features=build_pair_features(conv_vectors, tool_vectors),
        labels=np.asarray([example.label for example in examples], dtype="float32"),
        conversation_ids=[example.conversation_id for example in examples],
        tool_ids=[example.tool_id for example in examples],
        raw_scores=np.asarray([example.raw_score for example in examples], dtype="float32"),
    )


def encode_sequence_examples(
    examples: list[PairExample],
    encoder: Any,
    config: dict[str, Any],
    *,
    split_name: str,
    lexical_builder: LexicalFeatureBuilder | None = None,
    policy_builder: PolicyFeatureBuilder | None = None,
    gate_builder: PolicyFeatureBuilder | None = None,
    policy_as_lexical: bool = True,
) -> EncodedSequencePairs:
    batch_size = int(config["encoder"].get("batch_size", 8))
    unique_conversations = sorted({example.conversation_id: example.conversation_text for example in examples}.items())
    unique_tools = sorted({example.tool_id: example.tool_text for example in examples}.items())
    conversation_ids = [item[0] for item in unique_conversations]
    tool_ids = [item[0] for item in unique_tools]
    conv_seq, conv_mask = encoder.encode_sequence(
        [item[1] for item in unique_conversations],
        batch_size=batch_size,
        cache_key=f"{split_name}_conversations",
    )
    tool_seq, tool_mask = encoder.encode_sequence(
        [item[1] for item in unique_tools],
        batch_size=batch_size,
        cache_key=f"{split_name}_tools",
    )
    conv_index = {conversation_id: index for index, conversation_id in enumerate(conversation_ids)}
    tool_index = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    lexical_features = None
    if lexical_builder is not None:
        lexical_features = lexical_builder.transform(
            [example.conversation_text for example in examples],
            [example.tool_text for example in examples],
        )
    if policy_builder is not None:
        policy_features = policy_builder.transform(examples)
        if policy_as_lexical:
            lexical_features = (
                policy_features
                if lexical_features is None
                else np.concatenate([lexical_features, policy_features], axis=1).astype("float32")
            )
        else:
            policy_features = policy_features.astype("float32")
    else:
        policy_features = None
    gate_features = gate_builder.transform(examples).astype("float32") if gate_builder is not None else None
    conv_indices = np.asarray([conv_index[example.conversation_id] for example in examples], dtype=np.int64)
    tool_indices = np.asarray([tool_index[example.tool_id] for example in examples], dtype=np.int64)
    return EncodedSequencePairs(
        conv_sequences=np.asarray(conv_seq[conv_indices], dtype="float32"),
        conv_masks=np.asarray(conv_mask[conv_indices], dtype="bool"),
        tool_sequences=np.asarray(tool_seq[tool_indices], dtype="float32"),
        tool_masks=np.asarray(tool_mask[tool_indices], dtype="bool"),
        lexical_features=lexical_features,
        policy_features=policy_features if not policy_as_lexical else None,
        gate_features=gate_features,
        labels=np.asarray([example.label for example in examples], dtype="float32"),
        axis_labels=_axis_labels_array(examples),
        conversation_ids=[example.conversation_id for example in examples],
        tool_ids=[example.tool_id for example in examples],
        raw_scores=np.asarray([example.raw_score for example in examples], dtype="float32"),
    )


def encode_field_sequence_examples(
    examples: list[PairExample],
    encoder: Any,
    config: dict[str, Any],
    *,
    split_name: str,
    field_names: list[str],
    lexical_builder: LexicalFeatureBuilder | None = None,
    policy_builder: PolicyFeatureBuilder | None = None,
) -> EncodedFieldSequencePairs:
    batch_size = int(config["encoder"].get("batch_size", 8))
    unique_conversations = sorted({example.conversation_id: example.conversation_text for example in examples}.items())
    unique_tools = sorted({example.tool_id: example.tool_fields for example in examples}.items())
    conversation_ids = [item[0] for item in unique_conversations]
    tool_ids = [item[0] for item in unique_tools]
    conv_seq, conv_mask = encoder.encode_sequence(
        [item[1] for item in unique_conversations],
        batch_size=batch_size,
        cache_key=f"{split_name}_conversations",
    )
    field_sequences = {}
    field_masks = {}
    for field_name in field_names:
        seq, mask = encoder.encode_sequence(
            [fields[field_name] for _, fields in unique_tools],
            batch_size=batch_size,
            cache_key=f"{split_name}_tools_{field_name}",
        )
        field_sequences[field_name] = seq
        field_masks[field_name] = mask

    conv_index = {conversation_id: index for index, conversation_id in enumerate(conversation_ids)}
    tool_index = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    lexical_features = None
    if lexical_builder is not None:
        lexical_features = lexical_builder.transform(
            [example.conversation_text for example in examples],
            [example.tool_text for example in examples],
        )
    if policy_builder is not None:
        policy_features = policy_builder.transform(examples)
        lexical_features = (
            policy_features
            if lexical_features is None
            else np.concatenate([lexical_features, policy_features], axis=1).astype("float32")
        )
    conv_indices = np.asarray([conv_index[example.conversation_id] for example in examples], dtype=np.int64)
    tool_indices = np.asarray([tool_index[example.tool_id] for example in examples], dtype=np.int64)
    return EncodedFieldSequencePairs(
        field_names=field_names,
        conv_sequences=np.asarray(conv_seq[conv_indices], dtype="float32"),
        conv_masks=np.asarray(conv_mask[conv_indices], dtype="bool"),
        field_sequences={
            field_name: np.asarray(field_sequences[field_name][tool_indices], dtype="float32")
            for field_name in field_names
        },
        field_masks={
            field_name: np.asarray(field_masks[field_name][tool_indices], dtype="bool")
            for field_name in field_names
        },
        lexical_features=lexical_features,
        labels=np.asarray([example.label for example in examples], dtype="float32"),
        conversation_ids=[example.conversation_id for example in examples],
        tool_ids=[example.tool_id for example in examples],
        raw_scores=np.asarray([example.raw_score for example in examples], dtype="float32"),
    )


def build_tensor_dataset(pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs) -> TensorDataset:
    if isinstance(pairs, EncodedFieldSequencePairs):
        tensors = [
            torch.from_numpy(pairs.conv_sequences),
            torch.from_numpy(pairs.conv_masks),
        ]
        for field_name in pairs.field_names:
            tensors.extend(
                [
                    torch.from_numpy(pairs.field_sequences[field_name]),
                    torch.from_numpy(pairs.field_masks[field_name]),
                ]
            )
        if pairs.lexical_features is not None:
            tensors.append(torch.from_numpy(pairs.lexical_features))
        tensors.append(torch.from_numpy(pairs.labels))
        return TensorDataset(*tensors)
    if isinstance(pairs, EncodedSequencePairs):
        if (
            pairs.lexical_features is not None
            or pairs.policy_features is not None
            or pairs.gate_features is not None
        ):
            count = len(pairs.labels)
            return TensorDataset(
                torch.from_numpy(pairs.conv_sequences),
                torch.from_numpy(pairs.conv_masks),
                torch.from_numpy(pairs.tool_sequences),
                torch.from_numpy(pairs.tool_masks),
                torch.from_numpy(_feature_or_empty(pairs.lexical_features, count)),
                torch.from_numpy(_feature_or_empty(pairs.policy_features, count)),
                torch.from_numpy(_feature_or_empty(pairs.gate_features, count)),
                torch.from_numpy(pairs.labels),
                torch.from_numpy(_axis_labels_or_empty(pairs.axis_labels, count)),
            )
        return TensorDataset(
            torch.from_numpy(pairs.conv_sequences),
            torch.from_numpy(pairs.conv_masks),
            torch.from_numpy(pairs.tool_sequences),
            torch.from_numpy(pairs.tool_masks),
            torch.from_numpy(pairs.labels),
            torch.from_numpy(_axis_labels_or_empty(pairs.axis_labels, len(pairs.labels))),
        )
    return TensorDataset(
        torch.from_numpy(pairs.features),
        torch.from_numpy(pairs.labels),
    )


def train_one_epoch(
    *,
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    full_train: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    component_sums: dict[str, float] = {}
    component_counts: dict[str, int] = {}
    criterion = torch.nn.SmoothL1Loss()
    grad_accum_steps = max(1, int(config["train"].get("grad_accum_steps", 1)))
    loss_cfg = config.get("loss", {})
    pointwise_weight = float(loss_cfg.get("pointwise_weight", 1.0))
    optimizer.zero_grad()
    for step, batch in enumerate(loader, start=1):
        outputs, labels, axis_labels = model_batch_outputs(model, batch, device)
        predictions = _final_prediction(outputs)
        pointwise_loss = criterion(predictions, labels)
        loss = pointwise_weight * pointwise_loss
        if isinstance(model, FactorizedLateInteractionRegressor):
            if axis_labels is None or axis_labels.shape[1] == 0:
                raise ValueError("factorized model requires axis_labels in the dataset")
            factor_loss = criterion(outputs["factors"], axis_labels)
            factor_weighted_loss = float(loss_cfg.get("factor_weight", 0.3)) * factor_loss
            loss = loss + factor_weighted_loss
            _accumulate_loss(component_sums, component_counts, "factor_loss", factor_loss)
            _accumulate_loss(component_sums, component_counts, "factor_weighted_loss", factor_weighted_loss)
        (loss / grad_accum_steps).backward()
        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        _accumulate_loss(component_sums, component_counts, "pointwise_loss", pointwise_loss)
        _accumulate_loss(component_sums, component_counts, "pointwise_weighted_loss", loss)
    if len(loader) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    pairwise_weight = float(config["train"].get("pairwise_weight", 0.0))
    ranking_weights = {
        "pairwise": pairwise_weight,
        "listnet": float(loss_cfg.get("listnet_weight", 0.0)),
        "lambda_ndcg": float(loss_cfg.get("lambda_ndcg_weight", 0.0)),
        "bad_exposure": float(loss_cfg.get("bad_exposure_weight", 0.0)),
        "no_tool_fp": float(loss_cfg.get("no_tool_fp_weight", 0.0)),
        "score_spread": float(loss_cfg.get("score_spread_weight", 0.0)),
    }
    if any(weight > 0 for weight in ranking_weights.values()):
        labels = torch.from_numpy(full_train.labels).to(device)
        predictions = predict_all(model, full_train, device)
        ranking_loss = predictions.sum() * 0.0
        if ranking_weights["pairwise"] > 0:
            component = pairwise_margin_loss(
                predictions=predictions,
                labels=labels,
                conversation_ids=full_train.conversation_ids,
                margin=float(config["train"].get("pairwise_margin", 0.1)),
            )
            ranking_loss = ranking_loss + ranking_weights["pairwise"] * component
            _accumulate_loss(component_sums, component_counts, "pairwise_loss", component)
            _accumulate_loss(
                component_sums,
                component_counts,
                "pairwise_weighted_loss",
                ranking_weights["pairwise"] * component,
            )
        if ranking_weights["listnet"] > 0:
            component = listnet_loss(
                predictions=predictions,
                labels=labels,
                conversation_ids=full_train.conversation_ids,
                temperature=float(loss_cfg.get("listnet_temperature", 0.7)),
                min_label_gap=float(loss_cfg.get("listnet_min_label_gap", 0.05)),
            )
            ranking_loss = ranking_loss + ranking_weights["listnet"] * component
            _accumulate_loss(component_sums, component_counts, "listnet_loss", component)
            _accumulate_loss(
                component_sums,
                component_counts,
                "listnet_weighted_loss",
                ranking_weights["listnet"] * component,
            )
        if ranking_weights["lambda_ndcg"] > 0:
            component = lambda_ndcg_loss(
                predictions=predictions,
                labels=labels,
                conversation_ids=full_train.conversation_ids,
                topk=int(loss_cfg.get("lambda_ndcg_topk", 5)),
                sigma=float(loss_cfg.get("lambda_ndcg_sigma", 1.0)),
                min_label_gap=float(loss_cfg.get("lambda_ndcg_min_label_gap", 0.05)),
            )
            ranking_loss = ranking_loss + ranking_weights["lambda_ndcg"] * component
            _accumulate_loss(component_sums, component_counts, "lambda_ndcg_loss", component)
            _accumulate_loss(
                component_sums,
                component_counts,
                "lambda_ndcg_weighted_loss",
                ranking_weights["lambda_ndcg"] * component,
            )
        if ranking_weights["bad_exposure"] > 0:
            component = bad_exposure_loss(
                predictions=predictions,
                labels=labels,
                bad_label_threshold=float(loss_cfg.get("bad_label_threshold", -1.0)),
                bad_score_margin=float(loss_cfg.get("bad_score_margin", 0.0)),
            )
            ranking_loss = ranking_loss + ranking_weights["bad_exposure"] * component
            _accumulate_loss(component_sums, component_counts, "bad_exposure_loss", component)
            _accumulate_loss(
                component_sums,
                component_counts,
                "bad_exposure_weighted_loss",
                ranking_weights["bad_exposure"] * component,
            )
        if ranking_weights["no_tool_fp"] > 0:
            component = no_tool_false_positive_loss(
                predictions=predictions,
                labels=labels,
                conversation_ids=full_train.conversation_ids,
                high_label_threshold=float(loss_cfg.get("high_label_threshold", 1.0)),
                high_score_threshold=float(loss_cfg.get("high_score_threshold", 1.0)),
            )
            ranking_loss = ranking_loss + ranking_weights["no_tool_fp"] * component
            _accumulate_loss(component_sums, component_counts, "no_tool_fp_loss", component)
            _accumulate_loss(
                component_sums,
                component_counts,
                "no_tool_fp_weighted_loss",
                ranking_weights["no_tool_fp"] * component,
            )
        if ranking_weights["score_spread"] > 0:
            component = score_spread_loss(
                predictions=predictions,
                labels=labels,
                conversation_ids=full_train.conversation_ids,
                target_std=float(loss_cfg.get("score_spread_target_std", 0.7)),
            )
            ranking_loss = ranking_loss + ranking_weights["score_spread"] * component
            _accumulate_loss(component_sums, component_counts, "score_spread_loss", component)
            _accumulate_loss(
                component_sums,
                component_counts,
                "score_spread_weighted_loss",
                ranking_weights["score_spread"] * component,
            )
        optimizer.zero_grad()
        ranking_loss.backward()
        optimizer.step()
        _accumulate_loss(component_sums, component_counts, "ranking_weighted_loss", ranking_loss)

    metrics = {
        name: value / max(1, component_counts[name])
        for name, value in sorted(component_sums.items())
    }
    metrics["train_loss"] = metrics.get("pointwise_weighted_loss", 0.0) + metrics.get("ranking_weighted_loss", 0.0)
    return metrics


def evaluate_saved_checkpoints(
    *,
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    checkpoint_dir: Path,
    train_pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    val_pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    include_train_metrics: bool = True,
) -> None:
    primary_objective = str(config["train"].get("checkpoint_objective", "composite"))
    best_states = init_best_states(primary_objective)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    checkpoint_paths = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoint_paths:
        raise RuntimeError(f"no checkpoints found in {checkpoint_dir}")

    train_losses = _read_train_losses(run_dir / "train_log.jsonl")
    best_checkpoint_paths: dict[str, Path] = {}
    for checkpoint_path in checkpoint_paths:
        epoch = _epoch_from_checkpoint_path(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        train_metrics = evaluate(model, train_pairs, config, device) if include_train_metrics else {}
        val_metrics = evaluate(model, val_pairs, config, device)
        row = {
            "epoch": epoch,
            "train_loss": train_losses.get(epoch),
            "checkpoint": str(checkpoint_path),
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(metrics_path, row)
        print(
            f"[eval epoch {epoch:03d}] "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_top1={val_metrics['top1_match']:.3f} "
            f"val_top3={val_metrics['topk_recall']:.3f} "
            f"val_ndcg5={val_metrics.get('ndcg_at_5', 0.0):.3f} "
            f"val_regret5={val_metrics.get('top5_regret', 0.0):.3f} "
            f"val_bad3={val_metrics.get('bad_top3_rate', 0.0):.3f}"
        )
        changed = update_best_states(states=best_states, row=row)
        for name in changed:
            best_checkpoint_paths[name] = checkpoint_path

    materialize_best_checkpoints(
        model=model,
        states=best_states,
        checkpoint_paths=best_checkpoint_paths,
        pairs=val_pairs,
        run_dir=run_dir,
        device=device,
        primary_objective=primary_objective,
    )
    write_json(run_dir / "report.json", build_report(best_states, primary_objective=primary_objective))


def _accumulate_loss(
    sums: dict[str, float],
    counts: dict[str, int],
    name: str,
    value: torch.Tensor,
) -> None:
    sums[name] = sums.get(name, 0.0) + float(value.detach().cpu())
    counts[name] = counts.get(name, 0) + 1


def init_best_states(primary_objective: str) -> dict[str, dict[str, Any]]:
    objectives = {
        "legacy_score": {"direction": "max"},
        "composite": {"direction": "max"},
        "ndcg_at_5": {"direction": "max"},
        "top5_regret": {"direction": "min"},
        "safety": {"direction": "max"},
    }
    if primary_objective not in objectives:
        raise ValueError(f"unknown checkpoint_objective: {primary_objective}")
    return {
        name: {
            "direction": spec["direction"],
            "score": None,
            "row": {},
        }
        for name, spec in objectives.items()
    }


def update_best_states(
    *,
    states: dict[str, dict[str, Any]],
    row: dict[str, Any],
) -> list[str]:
    scores = checkpoint_scores(row["val"])
    changed = []
    for name, score in scores.items():
        state = states[name]
        if _is_better(score, current=state["score"], direction=str(state["direction"])):
            state["score"] = score
            state["row"] = row
            changed.append(name)
    return changed


def materialize_best_checkpoints(
    *,
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    states: dict[str, dict[str, Any]],
    checkpoint_paths: dict[str, Path],
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    run_dir: Path,
    device: torch.device,
    primary_objective: str,
) -> None:
    for name in states:
        checkpoint_path = checkpoint_paths.get(name)
        if checkpoint_path is None:
            continue
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        torch.save(model.state_dict(), run_dir / f"best_by_{name}.pt")
        write_predictions(run_dir / f"best_by_{name}_val_predictions.jsonl", model, pairs, device)
        if name == primary_objective:
            torch.save(model.state_dict(), run_dir / "best.pt")
            write_predictions(run_dir / "best_val_predictions.jsonl", model, pairs, device)


def update_best_checkpoints(
    *,
    states: dict[str, dict[str, Any]],
    row: dict[str, Any],
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    run_dir: Path,
    device: torch.device,
    primary_objective: str,
) -> None:
    changed = update_best_states(states=states, row=row)
    for name in changed:
        torch.save(model.state_dict(), run_dir / f"best_by_{name}.pt")
        write_predictions(run_dir / f"best_by_{name}_val_predictions.jsonl", model, pairs, device)
        if name == primary_objective:
            torch.save(model.state_dict(), run_dir / "best.pt")
            write_predictions(run_dir / "best_val_predictions.jsonl", model, pairs, device)


def checkpoint_scores(metrics: dict[str, float]) -> dict[str, float]:
    legacy = metrics["topk_recall"] - metrics["no_tool_fp_rate"] - metrics["mae"]
    composite = (
        0.35 * metrics.get("ndcg_at_5", 0.0)
        + 0.25 * metrics["topk_recall"]
        + 0.20 * metrics.get("high_value_recall_at_5", 0.0)
        - 0.20 * metrics.get("top5_regret", 0.0)
        - 0.20 * metrics.get("bad_top3_rate", 0.0)
        - 0.10 * metrics["mae"]
    )
    safety = (
        0.35 * metrics.get("high_conf_precision", 0.0)
        + 0.25 * (1.0 - metrics.get("bad_top3_rate", 0.0))
        + 0.20 * (1.0 - metrics.get("bad_top5_rate", 0.0))
        + 0.20 * (1.0 - metrics["no_tool_fp_rate"])
    )
    return {
        "legacy_score": float(legacy),
        "composite": float(composite),
        "ndcg_at_5": float(metrics.get("ndcg_at_5", 0.0)),
        "top5_regret": float(metrics.get("top5_regret", 0.0)),
        "safety": float(safety),
    }


def build_report(
    states: dict[str, dict[str, Any]],
    *,
    primary_objective: str,
) -> dict[str, Any]:
    best_by = {
        name: {
            "score": state["score"],
            "direction": state["direction"],
            "row": state["row"],
        }
        for name, state in states.items()
    }
    primary = best_by[primary_objective]
    return {
        "checkpoint_objective": primary_objective,
        "best_score": primary["score"],
        "best": primary["row"],
        "best_by": best_by,
    }


def _is_better(score: float, *, current: float | None, direction: str) -> bool:
    if current is None:
        return True
    if direction == "max":
        return score > current
    if direction == "min":
        return score < current
    raise ValueError(f"unknown objective direction: {direction}")


def _read_train_losses(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    losses: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        losses[int(row["epoch"])] = float(row["train_loss"])
    return losses


def _epoch_from_checkpoint_path(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def predict_batch(
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    outputs, labels, axis_labels = model_batch_outputs(model, batch, device)
    return _final_prediction(outputs), labels, axis_labels


def model_batch_outputs(
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    if len(batch) == 2:
        features, labels = batch
        return model(features.to(device)), labels.to(device), None
    if isinstance(model, FieldInteractionRegressor):
        conv_seq = batch[0]
        conv_mask = batch[1]
        offset = 2
        field_sequences = {}
        field_masks = {}
        for field_name in model.field_names:
            field_sequences[field_name] = batch[offset].to(device)
            field_masks[field_name] = batch[offset + 1].to(device)
            offset += 2
        lexical_features = None
        if len(batch) == 2 + len(model.field_names) * 2 + 2:
            lexical_features = batch[offset].to(device)
            offset += 1
        labels = batch[offset]
        return (
            model(
                conv_seq.to(device),
                conv_mask.to(device),
                field_sequences,
                field_masks,
                lexical_features,
            ),
            labels.to(device),
            None,
        )
    if len(batch) == 9:
        (
            conv_seq,
            conv_mask,
            tool_seq,
            tool_mask,
            lexical_features,
            policy_features,
            gate_features,
            labels,
            axis_labels,
        ) = batch
        return (
            model(
                conv_seq.to(device),
                conv_mask.to(device),
                tool_seq.to(device),
                tool_mask.to(device),
                _none_if_empty(lexical_features.to(device)),
                _none_if_empty(policy_features.to(device)),
                _none_if_empty(gate_features.to(device)),
            ),
            labels.to(device),
            _none_if_empty(axis_labels.to(device)),
        )
    if len(batch) == 8:
        conv_seq, conv_mask, tool_seq, tool_mask, lexical_features, policy_features, gate_features, labels = batch
        return (
            model(
                conv_seq.to(device),
                conv_mask.to(device),
                tool_seq.to(device),
                tool_mask.to(device),
                _none_if_empty(lexical_features.to(device)),
                _none_if_empty(policy_features.to(device)),
                _none_if_empty(gate_features.to(device)),
            ),
            labels.to(device),
            None,
        )
    if len(batch) == 6:
        conv_seq, conv_mask, tool_seq, tool_mask, labels, axis_labels = batch
        return (
            model(
                conv_seq.to(device),
                conv_mask.to(device),
                tool_seq.to(device),
                tool_mask.to(device),
            ),
            labels.to(device),
            _none_if_empty(axis_labels.to(device)),
        )
    conv_seq, conv_mask, tool_seq, tool_mask, labels = batch
    return (
        model(
            conv_seq.to(device),
            conv_mask.to(device),
            tool_seq.to(device),
            tool_mask.to(device),
        ),
        labels.to(device),
        None,
    )


def predict_all(
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(pairs, EncodedFieldSequencePairs):
        outputs = []
        batch_size = 256
        for start in range(0, len(pairs.labels), batch_size):
            end = min(start + batch_size, len(pairs.labels))
            lexical_features = (
                torch.from_numpy(pairs.lexical_features[start:end]).to(device)
                if pairs.lexical_features is not None
                else None
            )
            outputs.append(
                _final_prediction(model(
                    torch.from_numpy(pairs.conv_sequences[start:end]).to(device),
                    torch.from_numpy(pairs.conv_masks[start:end]).to(device),
                    {
                        field_name: torch.from_numpy(pairs.field_sequences[field_name][start:end]).to(device)
                        for field_name in pairs.field_names
                    },
                    {
                        field_name: torch.from_numpy(pairs.field_masks[field_name][start:end]).to(device)
                        for field_name in pairs.field_names
                    },
                    lexical_features,
                ))
            )
        return torch.cat(outputs, dim=0)
    if isinstance(pairs, EncodedSequencePairs):
        outputs = []
        batch_size = 256
        for start in range(0, len(pairs.labels), batch_size):
            end = min(start + batch_size, len(pairs.labels))
            lexical_features = (
                torch.from_numpy(pairs.lexical_features[start:end]).to(device)
                if pairs.lexical_features is not None
                else None
            )
            policy_features = (
                torch.from_numpy(pairs.policy_features[start:end]).to(device)
                if pairs.policy_features is not None
                else None
            )
            gate_features = (
                torch.from_numpy(pairs.gate_features[start:end]).to(device)
                if pairs.gate_features is not None
                else None
            )
            outputs.append(
                _final_prediction(model(
                    torch.from_numpy(pairs.conv_sequences[start:end]).to(device),
                    torch.from_numpy(pairs.conv_masks[start:end]).to(device),
                    torch.from_numpy(pairs.tool_sequences[start:end]).to(device),
                    torch.from_numpy(pairs.tool_masks[start:end]).to(device),
                    lexical_features,
                    policy_features,
                    gate_features,
                ))
            )
        return torch.cat(outputs, dim=0)
    return _final_prediction(model(torch.from_numpy(pairs.features).to(device)))


def pairwise_margin_loss(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    conversation_ids: list[str],
    margin: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    by_group: dict[str, list[int]] = {}
    for index, conversation_id in enumerate(conversation_ids):
        by_group.setdefault(conversation_id, []).append(index)
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, device=predictions.device)
        group_predictions = predictions[idx]
        group_labels = labels[idx]
        best = int(torch.argmax(group_labels))
        worst = int(torch.argmin(group_labels))
        if float(group_labels[best] - group_labels[worst]) < 0.05:
            continue
        losses.append(torch.relu(margin - (group_predictions[best] - group_predictions[worst])))
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


def listnet_loss(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    conversation_ids: list[str],
    temperature: float,
    min_label_gap: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    by_group: dict[str, list[int]] = {}
    for index, conversation_id in enumerate(conversation_ids):
        by_group.setdefault(conversation_id, []).append(index)
    temperature = max(temperature, 1e-4)
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, device=predictions.device)
        group_predictions = predictions[idx]
        group_labels = labels[idx]
        if float(group_labels.max() - group_labels.min()) < min_label_gap:
            continue
        target = torch.softmax(group_labels / temperature, dim=0)
        log_probs = torch.log_softmax(group_predictions / temperature, dim=0)
        losses.append(-(target * log_probs).sum())
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


def lambda_ndcg_loss(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    conversation_ids: list[str],
    topk: int,
    sigma: float,
    min_label_gap: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    by_group = _group_indices(conversation_ids)
    sigma = max(sigma, 1e-4)
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, device=predictions.device)
        group_predictions = predictions[idx]
        group_labels = labels[idx]
        if float(group_labels.max() - group_labels.min()) < min_label_gap:
            continue
        ranks = _prediction_ranks(group_predictions)
        gains = _label_gains(group_labels)
        discounts = _rank_discounts(ranks, topk=topk)
        ideal_dcg = _ideal_dcg(gains, topk=topk).clamp_min(1e-6)

        label_diff = group_labels.unsqueeze(1) - group_labels.unsqueeze(0)
        ordered_pair_mask = label_diff > min_label_gap
        if not bool(ordered_pair_mask.any()):
            continue
        pred_diff = group_predictions.unsqueeze(1) - group_predictions.unsqueeze(0)
        gain_diff = torch.abs(gains.unsqueeze(1) - gains.unsqueeze(0))
        discount_diff = torch.abs(discounts.unsqueeze(1) - discounts.unsqueeze(0))
        delta_ndcg = gain_diff * discount_diff / ideal_dcg
        pair_loss = torch.nn.functional.softplus(-sigma * pred_diff) / sigma
        losses.append((delta_ndcg[ordered_pair_mask] * pair_loss[ordered_pair_mask]).mean())
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


def bad_exposure_loss(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    bad_label_threshold: float,
    bad_score_margin: float,
) -> torch.Tensor:
    bad_mask = labels <= bad_label_threshold
    if not bool(bad_mask.any()):
        return predictions.sum() * 0.0
    return torch.nn.functional.softplus(predictions[bad_mask] - bad_score_margin).mean()


def no_tool_false_positive_loss(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    conversation_ids: list[str],
    high_label_threshold: float,
    high_score_threshold: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for indices in _group_indices(conversation_ids).values():
        idx = torch.tensor(indices, device=predictions.device)
        group_labels = labels[idx]
        if float(group_labels.max()) >= high_label_threshold:
            continue
        group_predictions = predictions[idx]
        losses.append(torch.nn.functional.softplus(group_predictions.max() - high_score_threshold))
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


def score_spread_loss(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    conversation_ids: list[str],
    target_std: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for indices in _group_indices(conversation_ids).values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, device=predictions.device)
        group_labels = labels[idx]
        if float(group_labels.max() - group_labels.min()) < 0.05:
            continue
        group_predictions = predictions[idx]
        losses.append(torch.relu(target_std - group_predictions.std(unbiased=False)))
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


def _group_indices(conversation_ids: list[str]) -> dict[str, list[int]]:
    by_group: dict[str, list[int]] = {}
    for index, conversation_id in enumerate(conversation_ids):
        by_group.setdefault(conversation_id, []).append(index)
    return by_group


def _feature_or_empty(features: np.ndarray | None, count: int) -> np.ndarray:
    if features is None:
        return np.zeros((count, 0), dtype="float32")
    return features.astype("float32", copy=False)


def _axis_labels_array(examples: list[PairExample]) -> np.ndarray | None:
    axis_labels = [example.axis_labels for example in examples]
    if all(label is None for label in axis_labels):
        return None
    if any(label is None for label in axis_labels):
        missing = [
            f"{example.sample_id}/{example.tool_id}"
            for example, label in zip(examples, axis_labels, strict=True)
            if label is None
        ][:5]
        raise ValueError(f"axis labels are partially missing, examples={missing}")
    return np.asarray(axis_labels, dtype="float32")


def _axis_labels_or_empty(axis_labels: np.ndarray | None, count: int) -> np.ndarray:
    if axis_labels is None:
        return np.zeros((count, 0), dtype="float32")
    return axis_labels.astype("float32", copy=False)


def _none_if_empty(features: torch.Tensor) -> torch.Tensor | None:
    if features.shape[1] == 0:
        return None
    return features


def _final_prediction(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["final"]
    return outputs


def _prediction_ranks(group_predictions: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(group_predictions, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, len(order) + 1, device=group_predictions.device)
    return ranks.float()


def _label_gains(group_labels: torch.Tensor) -> torch.Tensor:
    shifted = group_labels - group_labels.min()
    return torch.clamp(shifted, min=0.0)


def _rank_discounts(ranks: torch.Tensor, *, topk: int) -> torch.Tensor:
    discounts = 1.0 / torch.log2(ranks + 1.0)
    return torch.where(ranks <= topk, discounts, torch.zeros_like(discounts))


def _ideal_dcg(gains: torch.Tensor, *, topk: int) -> torch.Tensor:
    k = min(topk, len(gains))
    ideal_gains = torch.sort(gains, descending=True).values[:k]
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=gains.device, dtype=gains.dtype))
    return torch.sum(ideal_gains * discounts)


@torch.no_grad()
def evaluate(
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = predict_all(model, pairs, device).cpu().numpy()
    eval_cfg = config["eval"]
    metrics = compute_metrics(
        labels=pairs.labels,
        predictions=predictions,
        conversation_ids=pairs.conversation_ids,
        tool_ids=pairs.tool_ids,
        high_label_threshold=float(eval_cfg.get("high_label_threshold", 0.8)),
        high_pred_threshold=float(eval_cfg.get("high_pred_threshold", 0.8)),
        topk=int(eval_cfg.get("topk", 3)),
    )
    if isinstance(model, FactorizedLateInteractionRegressor) and isinstance(pairs, EncodedSequencePairs):
        metrics.update(factor_metrics(model=model, pairs=pairs, device=device))
    return metrics


@torch.no_grad()
def write_predictions(
    path: Path,
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor | FactorizedLateInteractionRegressor,
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    device: torch.device,
) -> None:
    model.eval()
    predictions = predict_all(model, pairs, device).cpu().numpy()
    factor_predictions = None
    if isinstance(model, FactorizedLateInteractionRegressor) and isinstance(pairs, EncodedSequencePairs):
        factor_predictions = predict_factors(model, pairs, device).cpu().numpy()
    with path.open("w", encoding="utf-8") as handle:
        for index, pred in enumerate(predictions):
            row = {
                "conversation_id": pairs.conversation_ids[index],
                "tool_id": pairs.tool_ids[index],
                "label": float(pairs.labels[index]),
                "prediction": float(pred),
                "raw_score": float(pairs.raw_scores[index]),
            }
            if factor_predictions is not None:
                row["factor_predictions"] = [float(value) for value in factor_predictions[index]]
                if isinstance(pairs, EncodedSequencePairs) and pairs.axis_labels is not None:
                    row["factor_labels"] = [float(value) for value in pairs.axis_labels[index]]
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")


@torch.no_grad()
def predict_factors(
    model: FactorizedLateInteractionRegressor,
    pairs: EncodedSequencePairs,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    batch_size = 256
    for start in range(0, len(pairs.labels), batch_size):
        end = min(start + batch_size, len(pairs.labels))
        lexical_features = (
            torch.from_numpy(pairs.lexical_features[start:end]).to(device)
            if pairs.lexical_features is not None
            else None
        )
        policy_features = (
            torch.from_numpy(pairs.policy_features[start:end]).to(device)
            if pairs.policy_features is not None
            else None
        )
        gate_features = (
            torch.from_numpy(pairs.gate_features[start:end]).to(device)
            if pairs.gate_features is not None
            else None
        )
        batch_outputs = model(
            torch.from_numpy(pairs.conv_sequences[start:end]).to(device),
            torch.from_numpy(pairs.conv_masks[start:end]).to(device),
            torch.from_numpy(pairs.tool_sequences[start:end]).to(device),
            torch.from_numpy(pairs.tool_masks[start:end]).to(device),
            lexical_features,
            policy_features,
            gate_features,
        )
        outputs.append(batch_outputs["factors"])
    return torch.cat(outputs, dim=0)


def factor_metrics(
    *,
    model: FactorizedLateInteractionRegressor,
    pairs: EncodedSequencePairs,
    device: torch.device,
) -> dict[str, float]:
    if pairs.axis_labels is None:
        return {}
    predictions = predict_factors(model, pairs, device).cpu().numpy()
    labels = pairs.axis_labels
    error = predictions - labels
    metrics: dict[str, float] = {
        "factor_mae": float(np.mean(np.abs(error))),
        "factor_mse": float(np.mean(error * error)),
    }
    axis_names = [
        "capability_match",
        "action_demand",
        "target_specificity",
        "consent_boundary",
        "intervention_cost",
        "companionship_fit",
    ]
    for index, axis_name in enumerate(axis_names):
        axis_error = error[:, index]
        metrics[f"factor_{axis_name}_mae"] = float(np.mean(np.abs(axis_error)))
    return metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _default_field_names() -> list[str]:
    return ["identity", "description", "capabilities", "examples", "schema"]
